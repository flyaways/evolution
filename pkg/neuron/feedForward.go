package neuron

// FeedForward 单个神经元的前向传播
func (n *Neuron) FeedForward(inputs []float64) float64 {
	sum := n.Bias
	for i, input := range inputs {
		sum += input * n.Weights[i]
	}
	n.Output = Sigmoid(sum)
	return n.Output
}
