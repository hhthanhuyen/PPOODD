package henn

import (
	"image"
)

func Flatten[T any](v [][]T) []T {
	r := make([]T, 0)
	for _, row := range v {
		r = append(r, row...)
	}
	return r
}

func ArgMax(v []float64) int {
	idx := 0
	max := v[0]
	for i := 1; i < len(v); i++ {
		if max < v[i] {
			max = v[i]
			idx = i
		}
	}
	return idx
}

// NormalizeImage returns the normalized image as [][]float64.
// This assumes image is B & W, and calculates the average of RGB values.
func NormalizeImage(image image.Image) [][]float64 {
	bounds := image.Bounds()

	res := make([][]float64, bounds.Dy())
	for i := 0; i < bounds.Dy(); i++ {
		res[i] = make([]float64, bounds.Dx())
		for j := 0; j < bounds.Dx(); j++ {
			r, g, b, _ := image.At(j+bounds.Min.X, i+bounds.Min.Y).RGBA()
			res[i][j] = ((float64(r) + float64(g) + float64(b)) / 3) / 0xFFFF
		}
	}

	return res
}

func makeVec(slots int, v float64) []float64 {
	out := make([]float64, slots)
	for i := range out {
		out[i] = v
	}
	return out
}

func makeMask10(slots int) []float64 {
	out := make([]float64, slots)
	for i := 0; i < 10 && i < slots; i++ {
		out[i] = 1.0
	}
	return out
}
