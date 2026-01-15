package henn

import (
	"encoding/csv"
	"os"
	"strconv"
)

type TestSet struct {
	Image [][]float64
	Label int
}

var testSets []TestSet

const imageSize = 28
const testSetSize = 10000

func ReadAllTestCase(filepath string) []TestSet {
	if len(testSets) > 0 {
		return testSets
	}

	f, err := os.Open(filepath)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	rd := csv.NewReader(f)
	rows, _ := rd.ReadAll()

	testSets = make([]TestSet, 0, testSetSize)
	for _, row := range rows {
		label, err := strconv.Atoi(row[0])
		if err != nil {
			panic(err)
		}
		row = row[1:]

		image := make([][]float64, imageSize)
		for i := 0; i < imageSize; i++ {
			image[i] = make([]float64, imageSize)
			for j := 0; j < imageSize; j++ {
				image[i][j], err = strconv.ParseFloat(row[i*imageSize+j], 64)
				if err != nil {
					panic(err)
				}
				image[i][j] /= 255
			}
		}

		testSets = append(testSets, TestSet{Image: image, Label: label})
	}

	return testSets
}
