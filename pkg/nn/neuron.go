package nn

// Neuron structure
type Neuron struct {
	Weights []float64
	Bias    float64
}

// Forward pass for a neuron
func (n *Neuron) Forward(inputs []float64) float64 {
	sum := n.Bias
	for i, input := range inputs {
		sum += input * n.Weights[i]
	}
	return Sigmoid(sum)
}
