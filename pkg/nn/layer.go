package nn

// Layer structure
type Layer struct {
	neurons []Neuron
}

// Forward pass for a layer
func (l *Layer) Forward(inputs []float64) []float64 {
	outputs := make([]float64, len(l.neurons))
	for i, neuron := range l.neurons {
		outputs[i] = neuron.Forward(inputs)
	}
	return outputs
}
