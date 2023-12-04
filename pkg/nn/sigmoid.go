package nn

import (
	"math"
)

// Sigmoid 激活函数及其导数
func Sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func SigmoidDerivative(x float64) float64 {
	return x * (1.0 - x)
}
