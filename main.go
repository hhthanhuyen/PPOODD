package main

import (
	crand "crypto/rand"
	"encoding/binary"
	"flag"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"main/henn"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/tuneinsight/lattigo/v4/ckks"
	"github.com/tuneinsight/lattigo/v4/ring"
	"github.com/tuneinsight/lattigo/v4/rlwe"
	"github.com/tuneinsight/lattigo/v4/utils"
)

type App struct {
	Params ckks.Parameters
	Ctx    *henn.CKKSContext
	Model  *henn.HENeuralNet
}

func main() {
	mode := flag.String("mode", "one", "one|eval")

	imgPath := flag.String("img", "MNIST/testing/0/9951.jpg", "path to one image for -mode=one")
	mnistTrain := flag.String("mnist_train", "MNIST/training", "MNIST TRAIN root directory (expects subfolders 0..9) used to fit tau")
	mnistTest := flag.String("mnist_test", "MNIST/testing", "MNIST TEST root directory (expects subfolders 0..9) used to evaluate accuracy")
	notmnistRoot := flag.String("notmnist", "notMNIST_small", "NotMNIST root directory (expects subfolders A..J)")

	tau := flag.Float64("tau", 90000, "tau for homomorphic OOD gate; set NaN to refit tau from MNIST TRAIN (eval) or print raw dists (one)")
	tauQ := flag.Float64("tauq", 0.99, "quantile for refitting tau from MNIST TRAIN distances (used when -tau is NaN in eval)")

	kernel := flag.Int("k", 7, "kernel size used by EncryptIm2Col")
	stride := flag.Int("s", 3, "stride used by EncryptIm2Col")

	maxN := flag.Int("n", 0, "max images per split (0=no limit)")
	progressEvery := flag.Int("p", 200, "progress print every p images (0=silent)")

	logitThresh := flag.Float64("lth", 1e3, "logit threshold used by accept/reject rule after decrypt")

	sigmaWP := flag.Float64("sigma_wp", 0.0, "stddev of Gaussian noise added at PLAINTEXT level to gated logits; 0=disable")
	sigmaCP := flag.Float64("sigma_cp", 0.0, "stddev of Gaussian noise encoded with SMALL scale (=1) then added to ciphertext + Enc(0) rerand; 0=disable")

	workers := flag.Int("workers", 0, "number of parallel workers (0=auto)")
	flag.Parse()

	app := Setup()

	w := *workers
	if w <= 0 {
		w = runtime.NumCPU()
		if w < 1 {
			w = 1
		}
	}

	fmt.Printf("[workers] %d\n", w)
	fmt.Printf("[wp] sigma_wp=%.6f\n", *sigmaWP)
	fmt.Printf("[cp] sigma_cp=%.6f\n", *sigmaCP)

	switch *mode {
	case "one":
		r, err := HEInferOne(app, *imgPath, *kernel, *stride, *tau, *logitThresh, *sigmaWP, *sigmaCP)
		if err != nil {
			panic(err)
		}
		fmt.Printf("pred=%d accepted=%v\n", r.Pred, r.Accepted)
		fmt.Println("logits:", r.Logits)
		if len(r.Dists) > 0 {
			fmt.Println("dists :", r.Dists)
		}

	case "eval":
		tauUse := *tau
		if math.IsNaN(tauUse) {
			fmt.Printf("[calib] tau is NaN => collecting MNIST TRAIN distances to refit tau (q=%.3f)\n", *tauQ)

			trainPairs, err := listMNISTPairs(*mnistTrain, *maxN)
			if err != nil {
				panic(err)
			}

			bestDistsID, err := CollectBestDistsMNISTRawParallel(app, trainPairs, *kernel, *stride, w, *progressEvery)
			if err != nil {
				panic(err)
			}

			tauUse = ChooseTauFromID(bestDistsID, *tauQ)
			fmt.Printf("[calib] refit tau = %.6f (q=%.3f, N=%d)\n", tauUse, *tauQ, len(bestDistsID))
		} else {
			fmt.Printf("[eval] using provided tau = %.6f\n", tauUse)
		}

		rep, err := HEInferWithOODDParallel(app, *mnistTest, *notmnistRoot, *kernel, *stride, tauUse, *maxN, *progressEvery, *logitThresh, *sigmaWP, *sigmaCP, w)
		if err != nil {
			panic(err)
		}
		fmt.Printf("%+v\n", rep)

	default:
		panic("unknown -mode (use one|eval)")
	}
}

func Setup() *App {
	params, _ := ckks.NewParametersFromLiteral(henn.DefaultParams)

	// Client: keys generated ONCE
	ctx := henn.NewCKKSContext(params)

	// Server model
	model := henn.NewHENeuralNet(params, henn.DefaultLayers...)
	rots := model.Rotations()
	for c := 1; c < 10; c++ {
		rots = append(rots, -c)
	}

	ctx.GenRotationKeys(rots)
	model.Initialize(ctx.EvaluationKey)

	return &App{Params: params, Ctx: ctx, Model: model}
}

type OneResult struct {
	Pred     int
	Accepted bool
	Logits   []float64
	Dists    []float64
}

func HEInferOne(app *App, path string, kernel, stride int, tau float64, logitThresh float64, sigmaWP float64, sigmaCP float64) (OneResult, error) {
	img, err := readImage(path)
	if err != nil {
		return OneResult{}, err
	}

	x := henn.NormalizeImage(img)
	ctIn := app.Ctx.EncryptIm2Col(x, kernel, stride)

	ctLogits, ctSecond := app.Model.Infer(ctIn, tau)

	if !math.IsNaN(tau) {
		if sigmaWP > 0 {
			_ = AddWeightPrivacyNoise(app.Params, app.Ctx.Encoder, app.Ctx.Evaluator, ctLogits, 10, sigmaWP)
		}
		app.Ctx.Evaluator.MulRelin(ctLogits, ctSecond, ctLogits)
		app.Ctx.Evaluator.Rescale(ctLogits, app.Params.DefaultScale(), ctLogits)
		if sigmaCP > 0 {
			_ = AddCircuitPrivacy(app.Params, app.Ctx.Encoder, app.Ctx.Evaluator, app.Ctx.Encryptor, ctLogits, 10, sigmaCP)
		}
	}

	logits := app.Ctx.DecryptFloats(ctLogits, 10)

	pred := -1
	maxVal := math.Inf(-1)
	for i, v := range logits {
		if math.Abs(v) <= logitThresh {
			if v > maxVal {
				maxVal = v
				pred = i
			}
		}
	}

	accepted := (pred != -1)

	res := OneResult{Pred: pred, Accepted: accepted, Logits: logits}

	if math.IsNaN(tau) {
		res.Dists = app.Ctx.DecryptFloats(ctSecond, 10)
	}

	return res, nil
}

type HEGateReport struct {
	Tau        float64
	MaskThresh float64

	SigmaWP float64
	SigmaCP float64

	NID  int
	NOOD int

	AccID_All      float64
	AccID_Accepted float64
	RejectRateID   float64

	AcceptRateOOD float64
}

func HEInferWithOODDParallel(app *App, mnistTestRoot, notmnistRoot string, kernel, stride int, tau float64, maxN, progressEvery int, logitThresh float64, sigmaWP, sigmaCP float64, workers int) (HEGateReport, error) {
	mnistPairs, err := listMNISTPairs(mnistTestRoot, maxN)
	if err != nil {
		return HEGateReport{}, err
	}

	NID, accAll, accAccepted, rejectRate, err := evalMNISTPairsParallel(
		app, mnistPairs, kernel, stride, tau, logitThresh, sigmaWP, sigmaCP, workers, progressEvery,
	)
	if err != nil {
		return HEGateReport{}, err
	}

	notmnistPaths, err := listAllImages(notmnistRoot, maxN)
	if err != nil {
		return HEGateReport{}, err
	}

	NOOD, acceptRateOOD, err := evalOODPathsParallel(
		app, notmnistPaths, kernel, stride, tau, logitThresh, sigmaWP, sigmaCP, workers, progressEvery,
	)
	if err != nil {
		return HEGateReport{}, err
	}

	return HEGateReport{
		Tau:            tau,
		MaskThresh:     logitThresh,
		SigmaWP:        sigmaWP,
		SigmaCP:        sigmaCP,
		NID:            NID,
		NOOD:           NOOD,
		AccID_All:      accAll,
		AccID_Accepted: accAccepted,
		RejectRateID:   rejectRate,
		AcceptRateOOD:  acceptRateOOD,
	}, nil
}

type mnistPair struct {
	path  string
	label int
}

func evalMNISTPairsParallel(app *App, pairs []mnistPair, kernel, stride int, tau float64, logitThresh float64, sigmaWP, sigmaCP float64, workers int, progressEvery int) (N int, accAll float64, accAccepted float64, rejectRate float64, err error) {
	type job struct {
		path  string
		label int
	}
	type out struct {
		ok       bool
		accepted bool
		correct  bool
	}

	jobs := make(chan job, 2*workers)
	outs := make(chan out, 2*workers)

	var doneWorkers sync.WaitGroup
	doneWorkers.Add(workers)

	var processed int64

	for wid := 0; wid < workers; wid++ {
		go func() {
			defer doneWorkers.Done()

			localEnc := ckks.NewEncoder(app.Params)
			localEval := ckks.NewEvaluator(app.Params, app.Ctx.EvaluationKey)
			localEncryptor := ckks.NewEncryptor(app.Params, app.Ctx.SecretKey)
			localDecryptor := ckks.NewDecryptor(app.Params, app.Ctx.SecretKey)

			localModel := app.Model.CloneForWorker(app.Ctx.EvaluationKey)

			for j := range jobs {
				r, e := HEInferOneWorker(
					app.Params,
					localModel,
					localEnc,
					localEval,
					localEncryptor,
					localDecryptor,
					j.path,
					kernel,
					stride,
					tau,
					logitThresh,
					sigmaWP,
					sigmaCP,
				)
				if e != nil {
					outs <- out{ok: false}
					continue
				}

				outs <- out{ok: true, accepted: r.Accepted, correct: r.Pred == j.label}

				n := atomic.AddInt64(&processed, 1)
				if progressEvery > 0 && n%int64(progressEvery) == 0 {
					fmt.Printf("[ID/he-gate] done %d / %d\n", n, len(pairs))
				}
			}
		}()
	}

	go func() {
		for _, p := range pairs {
			jobs <- job{path: p.path, label: p.label}
		}
		close(jobs)
	}()

	go func() {
		doneWorkers.Wait()
		close(outs)
	}()

	total := 0
	correctAll := 0
	acceptedTotal := 0
	acceptedCorrect := 0
	rejected := 0

	for r := range outs {
		if !r.ok {
			continue
		}
		total++
		if r.correct {
			correctAll++
		}
		if r.accepted {
			acceptedTotal++
			if r.correct {
				acceptedCorrect++
			}
		} else {
			rejected++
		}
	}

	if total > 0 {
		accAll = float64(correctAll) / float64(total)
		rejectRate = float64(rejected) / float64(total)
	} else {
		accAll = math.NaN()
		rejectRate = math.NaN()
	}

	if acceptedTotal > 0 {
		accAccepted = float64(acceptedCorrect) / float64(acceptedTotal)
	} else {
		accAccepted = math.NaN()
	}

	return total, accAll, accAccepted, rejectRate, nil
}

func evalOODPathsParallel(app *App, paths []string, kernel, stride int, tau float64, logitThresh float64, sigmaWP, sigmaCP float64, workers int, progressEvery int) (N int, acceptRate float64, err error) {
	type job struct{ path string }
	type out struct {
		ok       bool
		accepted bool
	}

	jobs := make(chan job, 2*workers)
	outs := make(chan out, 2*workers)

	var doneWorkers sync.WaitGroup
	doneWorkers.Add(workers)

	var processed int64

	for wid := 0; wid < workers; wid++ {
		go func() {
			defer doneWorkers.Done()

			localEnc := ckks.NewEncoder(app.Params)
			localEval := ckks.NewEvaluator(app.Params, app.Ctx.EvaluationKey)
			localEncryptor := ckks.NewEncryptor(app.Params, app.Ctx.SecretKey)
			localDecryptor := ckks.NewDecryptor(app.Params, app.Ctx.SecretKey)

			localModel := app.Model.CloneForWorker(app.Ctx.EvaluationKey)

			for j := range jobs {
				r, e := HEInferOneWorker(
					app.Params,
					localModel,
					localEnc,
					localEval,
					localEncryptor,
					localDecryptor,
					j.path,
					kernel,
					stride,
					tau,
					logitThresh,
					sigmaWP,
					sigmaCP,
				)
				if e != nil {
					outs <- out{ok: false}
					continue
				}

				outs <- out{ok: true, accepted: r.Accepted}

				n := atomic.AddInt64(&processed, 1)
				if progressEvery > 0 && n%int64(progressEvery) == 0 {
					fmt.Printf("[OOD/he-gate] done %d / %d\n", n, len(paths))
				}
			}
		}()
	}

	go func() {
		for _, p := range paths {
			jobs <- job{path: p}
		}
		close(jobs)
	}()

	go func() {
		doneWorkers.Wait()
		close(outs)
	}()

	total := 0
	accepted := 0
	for r := range outs {
		if !r.ok {
			continue
		}
		total++
		if r.accepted {
			accepted++
		}
	}

	if total > 0 {
		acceptRate = float64(accepted) / float64(total)
	} else {
		acceptRate = math.NaN()
	}

	return total, acceptRate, nil
}

func CollectBestDistsMNISTRawParallel(app *App, pairs []mnistPair, kernel, stride int, workers int, progressEvery int) ([]float64, error) {
	type job struct{ path string }
	type out struct {
		ok   bool
		best float64
	}

	jobs := make(chan job, 2*workers)
	outs := make(chan out, 2*workers)

	var doneWorkers sync.WaitGroup
	doneWorkers.Add(workers)

	var processed int64

	for wid := 0; wid < workers; wid++ {
		go func() {
			defer doneWorkers.Done()

			localEnc := ckks.NewEncoder(app.Params)
			localEncryptor := ckks.NewEncryptor(app.Params, app.Ctx.SecretKey)
			localDecryptor := ckks.NewDecryptor(app.Params, app.Ctx.SecretKey)
			localModel := app.Model.CloneForWorker(app.Ctx.EvaluationKey)

			for j := range jobs {
				best, e := bestDistOne(
					app.Params,
					localModel,
					localEnc,
					localEncryptor,
					localDecryptor,
					j.path,
					kernel,
					stride,
				)
				if e != nil {
					outs <- out{ok: false}
					continue
				}
				outs <- out{ok: true, best: best}

				n := atomic.AddInt64(&processed, 1)
				if progressEvery > 0 && n%int64(progressEvery) == 0 {
					fmt.Printf("[calib/ID] done %d / %d\n", n, len(pairs))
				}
			}
		}()
	}

	go func() {
		for _, p := range pairs {
			jobs <- job{path: p.path}
		}
		close(jobs)
	}()

	go func() {
		doneWorkers.Wait()
		close(outs)
	}()

	bestDists := make([]float64, 0, len(pairs))
	for r := range outs {
		if !r.ok {
			continue
		}
		bestDists = append(bestDists, r.best)
	}
	return bestDists, nil
}

func bestDistOne(params ckks.Parameters, model *henn.HENeuralNet, enc ckks.Encoder, encryptor rlwe.Encryptor, decryptor rlwe.Decryptor, path string, kernel, stride int) (float64, error) {
	img, err := readImage(path)
	if err != nil {
		return math.NaN(), err
	}
	x := henn.NormalizeImage(img)

	ctIn, err := encryptIm2ColLocal(params, enc, encryptor, x, kernel, stride)
	if err != nil {
		return math.NaN(), err
	}

	_, ctDists := model.Infer(ctIn, math.NaN())
	dists := decryptFloatsLocal(params, decryptor, enc, ctDists, 10)

	return minFloat64(dists), nil
}

func HEInferOneWorker(params ckks.Parameters, model *henn.HENeuralNet, enc ckks.Encoder, eval ckks.Evaluator, encryptor rlwe.Encryptor, decryptor rlwe.Decryptor, path string, kernel, stride int, tau float64, logitThresh float64, sigmaWP float64, sigmaCP float64) (OneResult, error) {
	img, err := readImage(path)
	if err != nil {
		return OneResult{}, err
	}
	x := henn.NormalizeImage(img)

	ctIn, err := encryptIm2ColLocal(params, enc, encryptor, x, kernel, stride)
	if err != nil {
		return OneResult{}, err
	}

	ctLogits, ctSecond := model.Infer(ctIn, tau)

	if !math.IsNaN(tau) {
		if sigmaWP > 0 {
			_ = AddWeightPrivacyNoise(params, enc, eval, ctLogits, 10, sigmaWP)
		}
		eval.MulRelin(ctLogits, ctSecond, ctLogits)
		eval.Rescale(ctLogits, params.DefaultScale(), ctLogits)
		if sigmaCP > 0 {
			_ = AddCircuitPrivacy(params, enc, eval, encryptor, ctLogits, 10, sigmaCP)
		}
	}

	logits := decryptFloatsLocal(params, decryptor, enc, ctLogits, 10)

	pred := -1
	maxVal := math.Inf(-1)
	for i, v := range logits {
		if math.Abs(v) <= logitThresh {
			if v > maxVal {
				maxVal = v
				pred = i
			}
		}
	}

	accepted := (pred != -1)

	return OneResult{Pred: pred, Accepted: accepted, Logits: logits}, nil
}

func decryptFloatsLocal(params ckks.Parameters, decryptor rlwe.Decryptor, enc ckks.Encoder, ct *rlwe.Ciphertext, n int) []float64 {
	pt := decryptor.DecryptNew(ct)
	c := enc.Decode(pt, params.LogSlots())
	out := make([]float64, n)
	for i := 0; i < n; i++ {
		out[i] = real(c[i])
	}
	return out
}

func encryptIm2ColLocal(params ckks.Parameters, enc ckks.Encoder, encryptor rlwe.Encryptor, img [][]float64, kernelSize int, stride int) (*rlwe.Ciphertext, error) {
	X := len(img)
	if X == 0 {
		return nil, fmt.Errorf("empty img")
	}
	Y := len(img[0])

	if (X-kernelSize+stride)%stride != 0 || (Y-kernelSize+stride)%stride != 0 {
		return nil, fmt.Errorf("size mismatch")
	}

	XX := kernelSize * kernelSize
	YY := ((X - kernelSize + stride) / stride) * ((Y - kernelSize + stride) / stride)

	encodedImg := make([][]float64, XX)
	for i := range encodedImg {
		encodedImg[i] = make([]float64, YY)
	}

	var xx, yy int
	for i := 0; i <= X-kernelSize; i += stride {
		for j := 0; j <= Y-kernelSize; j += stride {
			for ki := 0; ki < kernelSize; ki++ {
				for kj := 0; kj < kernelSize; kj++ {
					encodedImg[xx][yy] = img[i+ki][j+kj]
					xx++
				}
			}
			xx, yy = 0, yy+1
		}
	}

	flattened := make([]float64, 0, XX*YY)
	for _, row := range encodedImg {
		flattened = append(flattened, row...)
	}

	pt := enc.EncodeNew(flattened, params.MaxLevel(), params.DefaultScale(), params.LogSlots())
	return encryptor.EncryptNew(pt), nil
}

func ChooseTauFromID(bestDistsID []float64, q float64) float64 {
	if len(bestDistsID) == 0 {
		panic("ChooseTauFromID: empty ID list")
	}
	tmp := append([]float64(nil), bestDistsID...)
	sort.Float64s(tmp)
	return quantileSorted(tmp, q)
}

func quantileSorted(xs []float64, q float64) float64 {
	if len(xs) == 0 {
		panic("quantileSorted: empty slice")
	}
	if q <= 0 {
		return xs[0]
	}
	if q >= 1 {
		return xs[len(xs)-1]
	}
	pos := q * float64(len(xs)-1)
	i := int(math.Floor(pos))
	j := int(math.Ceil(pos))
	if i == j {
		return xs[i]
	}
	w := pos - float64(i)
	return xs[i]*(1.0-w) + xs[j]*w
}

func cryptoSeed() (int64, error) {
	var seedBytes [8]byte
	if _, err := crand.Read(seedBytes[:]); err != nil {
		return 0, err
	}
	return int64(binary.LittleEndian.Uint64(seedBytes[:])), nil
}

func AddWeightPrivacyNoise(params ckks.Parameters, enc ckks.Encoder, eval ckks.Evaluator, ct *rlwe.Ciphertext, K int, sigmaWP float64) error {
	if sigmaWP <= 0 || K <= 0 {
		return nil
	}
	seed, err := cryptoSeed()
	if err != nil {
		return err
	}
	rng := rand.New(rand.NewSource(seed))

	slots := params.Slots()
	noise := make([]float64, slots)
	for i := 0; i < K && i < slots; i++ {
		noise[i] = rng.NormFloat64() * sigmaWP
	}

	ptNoise := enc.EncodeNew(noise, ct.Level(), params.DefaultScale(), params.LogSlots())
	eval.Add(ct, ptNoise, ct)
	return nil
}

func AddCircuitPrivacy(params ckks.Parameters, enc ckks.Encoder, eval ckks.Evaluator, encryptor rlwe.Encryptor, ct *rlwe.Ciphertext, K int, sigmaCP float64) error {
	if sigmaCP <= 0 || K <= 0 {
		return nil
	}

	ringQ := params.RingQ()

	prng, err := utils.NewPRNG()
	if err != nil {
		return err
	}

	sampler := ring.NewGaussianSampler(prng, ringQ, sigmaCP, int(3*sigmaCP))
	e := ringQ.NewPoly()
	sampler.Read(e)
	if ct.IsNTT {
		ringQ.NTT(e, e)
	}
	ringQ.Add(ct.Value[0], e, ct.Value[0])

	// rerandomize by adding Enc(0)
	ct0 := encryptor.EncryptZeroNew(ct.Level())
	ct0.SetScale(ct.Scale)
	eval.Add(ct, ct0, ct)

	return nil
}

func listMNISTPairs(root string, maxN int) ([]mnistPair, error) {
	pairs := make([]mnistPair, 0, 1024)
	total := 0

	err := filepath.Walk(root, func(path string, info os.FileInfo, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if info.IsDir() || !isImageFile(path) {
			return nil
		}
		if maxN > 0 && total >= maxN {
			return nil
		}

		labelStr := filepath.Base(filepath.Dir(path))
		label, e := strconv.Atoi(labelStr)
		if e != nil || label < 0 || label > 9 {
			return nil
		}

		pairs = append(pairs, mnistPair{path: path, label: label})
		total++
		return nil
	})
	return pairs, err
}

func listAllImages(root string, maxN int) ([]string, error) {
	paths := make([]string, 0, 1024)
	total := 0

	err := filepath.Walk(root, func(path string, info os.FileInfo, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if info.IsDir() || !isImageFile(path) {
			return nil
		}
		if maxN > 0 && total >= maxN {
			return nil
		}
		paths = append(paths, path)
		total++
		return nil
	})
	return paths, err
}

func readImage(path string) (image.Image, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	return img, err
}

func isImageFile(path string) bool {
	ext := strings.ToLower(filepath.Ext(path))
	return ext == ".jpg" || ext == ".jpeg" || ext == ".png"
}

func minFloat64(xs []float64) float64 {
	if len(xs) == 0 {
		return math.Inf(1)
	}
	m := xs[0]
	for _, v := range xs[1:] {
		if v < m {
			m = v
		}
	}
	return m
}
