package neuron

import (
	"math/rand"
)

// NewNeuron 创建一个新的神经元
func NewNeuron(inputCount int) *Neuron {
	weights := make([]float64, inputCount)
	for i := range weights {
		weights[i] = rand.Float64()*2 - 1 // 随机初始化权重
	}
	return &Neuron{
		Weights: weights,
		Bias:    rand.Float64()*2 - 1, // 随机初始化偏置
	}
}
