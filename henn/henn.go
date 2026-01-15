package henn

import (
	"math"

	"github.com/tuneinsight/lattigo/v4/ckks"
	"github.com/tuneinsight/lattigo/v4/rlwe"
)

type HENeuralNet struct {
	Parameters    ckks.Parameters
	EvaluationKey rlwe.EvaluationKey
	Encoder       ckks.Encoder
	Evaluator     ckks.Evaluator
	Layers        []EncodedLayer
}

func NewHENeuralNet(params ckks.Parameters, layers ...Layer) *HENeuralNet {
	nn := &HENeuralNet{
		Parameters: params,
		Encoder:    ckks.NewEncoder(params),
		Evaluator:  nil,
	}
	nn.AddLayers(layers...)
	return nn
}

// Initialize initializes this neural network using sender's public evaluation keys.
func (nn *HENeuralNet) Initialize(evk rlwe.EvaluationKey) {
	nn.EvaluationKey = evk
	nn.Evaluator = ckks.NewEvaluator(nn.Parameters, evk)
}

func (nn *HENeuralNet) CloneForWorker(evk rlwe.EvaluationKey) *HENeuralNet {
	clone := &HENeuralNet{
		Parameters:    nn.Parameters,
		EvaluationKey: evk,
		Encoder:       ckks.NewEncoder(nn.Parameters),
		Evaluator:     ckks.NewEvaluator(nn.Parameters, evk),
		Layers:        nn.Layers,
	}
	return clone
}

// Rotations returns the number of rotations that are needed to infer from this neural network.
func (nn *HENeuralNet) Rotations() []int {
	rotSet := make(map[int]struct{})

	for _, l := range nn.Layers {
		switch l := l.(type) {
		case EncodedConvLayer:
			batchSize := l.Im2ColY
			N := l.Im2ColX

			for i, j := 0, N; j > 0; i, j = i+1, j>>1 {
				if j&1 == 1 {
					k := N - (N & ((2 << i) - 1))
					k *= batchSize
					rotSet[k] = struct{}{}
				}
				rotSet[batchSize*(1<<i)] = struct{}{}
			}

			for i := 0; i < len(l.Kernel); i++ {
				rotSet[-i*batchSize] = struct{}{}
			}

		case EncodedLinearLayer:
			for _, r := range l.Weights.Rotations() {
				rotSet[r] = struct{}{}
			}
		}
	}

	rot := make([]int, 0, len(rotSet))
	for k := range rotSet {
		rot = append(rot, k)
	}
	return rot
}

func (nn *HENeuralNet) AddLayers(layers ...Layer) {
	for _, l := range layers {
		switch l := l.(type) {
		case ConvLayer:
			nn.Layers = append(nn.Layers, nn.EncodeConvLayer(l))
		case LinearLayer:
			nn.Layers = append(nn.Layers, nn.EncodeLinearLayer(l))
		case ActivationLayer:
			nn.Layers = append(nn.Layers, l)
		}
	}
}

func (nn *HENeuralNet) Infer(ctIn *rlwe.Ciphertext, tau float64) (*rlwe.Ciphertext, *rlwe.Ciphertext) {
	if nn.Evaluator == nil {
		panic("model not initialized")
	}

	ctOut := ctIn.CopyNew()

	var done chan struct{}
	var ctDists *rlwe.Ciphertext
	var distErr error

	for idx, l := range nn.Layers {
		switch l := l.(type) {
		case EncodedConvLayer:
			nn.conv(l, ctOut)
		case EncodedLinearLayer:
			nn.linear(l, ctOut)
		case ActivationLayer:
			nn.activate(l, ctOut)
		}

		// Start Mahalanobis on features right after first layer output
		if idx == 0 {
			ctFeat := ctOut.CopyNew()
			done = make(chan struct{})

			go func() {
				localEval := ckks.NewEvaluator(nn.Parameters, nn.EvaluationKey)
				localEnc := ckks.NewEncoder(nn.Parameters)

				ctDists, distErr = nn.computeMahaDists(
					localEval,
					localEnc,
					ctFeat,
					means,
					precision,
					256, // feature dimension
				)

				if distErr == nil && !math.IsNaN(tau) {
					ctDists, distErr = nn.MaskFromDists(
						localEval,
						localEnc,
						ctDists,
						tau,
					)
				}

				close(done)
			}()
		}
	}

	if done != nil {
		<-done
		if distErr != nil {
			panic(distErr)
		}
	}

	return ctOut, ctDists
}

func (nn *HENeuralNet) EncodeConvLayer(cl ConvLayer) EncodedConvLayer {
	if len(cl.Kernel) != len(cl.Bias) {
		panic("dimension mismatch between kernel and bias")
	}

	kx := len(cl.Kernel[0])
	ky := len(cl.Kernel[0][0])
	kSize := kx * ky
	repeat := ((cl.InputX - kx + cl.Stride) / cl.Stride) * ((cl.InputY - ky + cl.Stride) / cl.Stride)

	encodedKernels := make([]*rlwe.Plaintext, len(cl.Kernel))
	for i, k := range cl.Kernel {
		flattenedKernel := make([]float64, 0, repeat*kSize)
		for ii := 0; ii < kx; ii++ {
			for jj := 0; jj < ky; jj++ {
				for n := 0; n < repeat; n++ {
					flattenedKernel = append(flattenedKernel, k[ii][jj])
				}
			}
		}
		encodedKernels[i] = nn.Encoder.EncodeNew(
			flattenedKernel,
			nn.Parameters.MaxLevel(),
			nn.Parameters.DefaultScale(),
			nn.Parameters.LogSlots(),
		)
	}

	encodedBiases := make([]*rlwe.Plaintext, len(cl.Bias))
	for i, b := range cl.Bias {
		repeatedBias := make([]float64, repeat)
		for j := range repeatedBias {
			repeatedBias[j] = b
		}
		encodedBiases[i] = nn.Encoder.EncodeNew(
			repeatedBias,
			nn.Parameters.MaxLevel(),
			nn.Parameters.DefaultScale(),
			nn.Parameters.LogSlots(),
		)
	}

	mask := make([]float64, repeat)
	for i := range mask {
		mask[i] = 1
	}
	encodedMask := nn.Encoder.EncodeNew(
		mask,
		nn.Parameters.MaxLevel(),
		nn.Parameters.DefaultScale(),
		nn.Parameters.LogSlots(),
	)

	return EncodedConvLayer{
		Im2ColX: kSize,
		Im2ColY: repeat,
		mask:    encodedMask,

		Kernel: encodedKernels,
		Bias:   encodedBiases,
		Stride: cl.Stride,
	}
}

func (nn *HENeuralNet) conv(cl EncodedConvLayer, ct *rlwe.Ciphertext) {
	ctConv := rlwe.NewCiphertext(nn.Parameters.Parameters, ct.Degree(), ct.Level())
	ctTemp := rlwe.NewCiphertext(nn.Parameters.Parameters, ct.Degree(), ct.Level())

	for i := range cl.Kernel {
		k := cl.Kernel[i]
		b := cl.Bias[i]

		nn.Evaluator.Mul(ct, k, ctTemp)
		nn.Evaluator.InnerSum(ctTemp, cl.Im2ColY, cl.Im2ColX, ctTemp)
		nn.Evaluator.Add(ctTemp, b, ctTemp)

		nn.Evaluator.Mul(ctTemp, cl.mask, ctTemp)
		nn.Evaluator.Rotate(ctTemp, -i*cl.Im2ColY, ctTemp)
		nn.Evaluator.Add(ctConv, ctTemp, ctConv)
	}

	ct.Copy(ctConv)
	nn.Evaluator.Rescale(ct, nn.Parameters.DefaultScale(), ct)
}

func (nn *HENeuralNet) EncodeLinearLayer(ll LinearLayer) EncodedLinearLayer {
	N := len(ll.Weights)
	M := len(ll.Weights[0])

	diagWeights := make(map[int][]float64, len(ll.Weights))
	slots := nn.Parameters.Slots()

	for i := 0; i < slots; i++ {
		isZero := true
		row := make([]float64, slots)

		for j := 0; j < slots; j++ {
			ii, jj := j%slots, (i+j)%slots
			if ii < N && jj < M {
				row[j] = ll.Weights[ii][jj]
				if row[j] != 0 {
					isZero = false
				}
			}
		}

		if !isZero {
			diagWeights[i] = row
		}
	}

	encodedWeights := ckks.GenLinearTransformBSGS(
		nn.Encoder,
		diagWeights,
		nn.Parameters.MaxLevel(),
		nn.Parameters.DefaultScale(),
		1.0,
		nn.Parameters.LogSlots(),
	)

	encodedBias := nn.Encoder.EncodeNew(
		ll.Bias,
		nn.Parameters.MaxLevel(),
		nn.Parameters.DefaultScale(),
		nn.Parameters.LogSlots(),
	)

	return EncodedLinearLayer{
		Weights: encodedWeights,
		Bias:    encodedBias,
	}
}

func (nn *HENeuralNet) linear(ll EncodedLinearLayer, ct *rlwe.Ciphertext) {
	nn.Evaluator.LinearTransform(ct, ll.Weights, []*rlwe.Ciphertext{ct})
	nn.Evaluator.Rescale(ct, nn.Parameters.DefaultScale(), ct)
	nn.Evaluator.Add(ct, ll.Bias, ct)
}

func (nn *HENeuralNet) activate(l ActivationLayer, ct *rlwe.Ciphertext) {
	l.ActivationFn(nn, ct)
}

func (nn *HENeuralNet) computeMahaDists(
	eval ckks.Evaluator,
	enc ckks.Encoder,
	ctFeat *rlwe.Ciphertext,
	means [][]float64,
	precision []float64,
	D int,
) (*rlwe.Ciphertext, error) {

	ctDist := rlwe.NewCiphertext(nn.Parameters.Parameters, ctFeat.Degree(), ctFeat.Level())
	level := ctFeat.Level()

	slots := nn.Parameters.Slots()
	mask0 := make([]float64, slots)
	mask0[0] = 1.0
	ptMask0 := enc.EncodeNew(mask0, level, nn.Parameters.DefaultScale(), nn.Parameters.LogSlots())
	ptPrec := enc.EncodeNew(precision, level, rlwe.NewScale(1), nn.Parameters.LogSlots())

	for c := 0; c < 10; c++ {
		ctTmp := ctFeat.CopyNew()
		ptMean := enc.EncodeNew(means[c], level, nn.Parameters.DefaultScale(), nn.Parameters.LogSlots())

		eval.Sub(ctTmp, ptMean, ctTmp)
		eval.MulRelin(ctTmp, ctTmp, ctTmp)
		eval.Rescale(ctTmp, nn.Parameters.DefaultScale(), ctTmp)

		eval.Mul(ctTmp, ptPrec, ctTmp)

		for step := 1; step < D; step <<= 1 {
			ctRot := eval.RotateNew(ctTmp, step)
			eval.Add(ctTmp, ctRot, ctTmp)
		}

		eval.Mul(ctTmp, ptMask0, ctTmp)
		eval.Rescale(ctTmp, nn.Parameters.DefaultScale(), ctTmp)

		if c > 0 {
			eval.Rotate(ctTmp, -c, ctTmp)
		}

		eval.Add(ctDist, ctTmp, ctDist)
	}

	return ctDist, nil
}

func (nn *HENeuralNet) MaskFromDists(
	eval ckks.Evaluator,
	enc ckks.Encoder,
	ctDist *rlwe.Ciphertext,
	tau float64,
) (*rlwe.Ciphertext, error) {

	params := nn.Parameters
	ctX := ctDist.CopyNew()
	eval.MultByConst(ctX, 1.0/tau, ctX)

	// repeated squaring (fixed 10 squarings)
	eval.MulRelin(ctX, ctX, ctX)
	eval.Rescale(ctX, params.DefaultScale(), ctX)
	eval.MulRelin(ctX, ctX, ctX)
	eval.Rescale(ctX, params.DefaultScale(), ctX)
	eval.MulRelin(ctX, ctX, ctX)
	eval.Rescale(ctX, params.DefaultScale(), ctX)
	eval.MulRelin(ctX, ctX, ctX)
	eval.Rescale(ctX, params.DefaultScale(), ctX)
	eval.MulRelin(ctX, ctX, ctX)
	eval.Rescale(ctX, params.DefaultScale(), ctX)

	eval.MulRelin(ctX, ctX, ctX)
	eval.Rescale(ctX, params.DefaultScale(), ctX)
	eval.MulRelin(ctX, ctX, ctX)
	eval.Rescale(ctX, params.DefaultScale(), ctX)
	eval.MulRelin(ctX, ctX, ctX)
	eval.Rescale(ctX, params.DefaultScale(), ctX)
	eval.MulRelin(ctX, ctX, ctX)
	eval.Rescale(ctX, params.DefaultScale(), ctX)
	eval.MulRelin(ctX, ctX, ctX)
	eval.Rescale(ctX, params.DefaultScale(), ctX)
	eval.AddConst(ctX, 1.0, ctX)

	return ctX, nil
}
