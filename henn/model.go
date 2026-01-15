package henn

import (
	_ "embed"

	"github.com/tuneinsight/lattigo/v4/ckks"
	"github.com/tuneinsight/lattigo/v4/rlwe"
)

var DefaultLayers []Layer

var DefaultParams = ckks.ParametersLiteral{
	LogN:     13,
	LogSlots: 12,

	LogQ: []int{
		53, 40, 40, 40, 40, 40, 40, 40, 40,
		40, 40, 40, 40, 40, 40, 40, 40, 40,
	},

	LogP:         []int{53, 53},
	DefaultScale: 1 << 40,
}

func init() {
	activationFn := func(model *HENeuralNet, ct *rlwe.Ciphertext) {
		model.Evaluator.MulRelin(ct, ct, ct)
		model.Evaluator.Rescale(ct, model.Parameters.DefaultScale(), ct)
	}

	DefaultLayers = []Layer{
		ConvLayer{
			InputX: 28,
			InputY: 28,
			Kernel: [][][]float64{convWeight0, convWeight1, convWeight2, convWeight3},
			Bias:   convBias,
			Stride: 3,
		},
		ActivationLayer{
			ActivationFn: activationFn,
		},
		LinearLayer{
			Weights: lin0Weight,
			Bias:    lin0Bias,
		},
		ActivationLayer{
			ActivationFn: activationFn,
		},
		LinearLayer{
			Weights: lin1Weight,
			Bias:    lin1Bias,
		},
	}
}
