package henn

import (
	"github.com/tuneinsight/lattigo/v4/ckks"
	"github.com/tuneinsight/lattigo/v4/rlwe"
)

type Layer interface {
	isLayer()
}

type ConvLayer struct {
	InputX int
	InputY int

	Kernel [][][]float64
	Bias   []float64
	Stride int
}

func (ConvLayer) isLayer() {}

type LinearLayer struct {
	Weights [][]float64
	Bias    []float64
}

func (LinearLayer) isLayer() {}

type ActivationLayer struct {
	ActivationFn func(*HENeuralNet, *rlwe.Ciphertext)
}

func (ActivationLayer) isLayer() {}

func (ActivationLayer) isEncodedLayer() {}

type EncodedLayer interface {
	isEncodedLayer()
}

type EncodedConvLayer struct {
	Im2ColX int // Same as kernel size
	Im2ColY int // Same as repeats
	mask    *rlwe.Plaintext

	Kernel []*rlwe.Plaintext
	Bias   []*rlwe.Plaintext
	Stride int
}

func (EncodedConvLayer) isEncodedLayer() {}

type EncodedLinearLayer struct {
	Weights ckks.LinearTransform
	Bias    *rlwe.Plaintext
}

func (EncodedLinearLayer) isEncodedLayer() {}
