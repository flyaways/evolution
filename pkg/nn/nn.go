package nn

import (
	"math/rand"
)

// NeuralNetwork structure
type NeuralNetwork struct {
	layers []Layer
}

// Forward pass for the network
func (nn *NeuralNetwork) Forward(inputs []float64) []float64 {
	currentInputs := inputs
	for _, layer := range nn.layers {
		currentInputs = layer.Forward(currentInputs)
	}
	return currentInputs
}

// Backpropagation and weight update
func (nn *NeuralNetwork) BackPropagate(inputs []float64, expected []float64, learnRate float64) {
	// Forward pass and store all layer outputs
	layerOutputs := [][]float64{inputs}
	for _, layer := range nn.layers {
		inputs = layer.Forward(inputs)
		layerOutputs = append(layerOutputs, inputs)
	}

	// Calculate the output error
	outputs := layerOutputs[len(layerOutputs)-1]
	outputErrors := make([]float64, len(outputs))
	for i, output := range outputs {
		outputErrors[i] = expected[i] - output
	}

	// Go through the layers backwards to calculate deltas and update Weights
	for i := len(nn.layers) - 1; i >= 0; i-- {
		layer := &nn.layers[i]
		inputs := layerOutputs[i]
		outputs := layerOutputs[i+1]
		deltas := make([]float64, len(layer.neurons))

		// Calculate the delta for the output layer
		if i == len(nn.layers)-1 {
			for j := range layer.neurons {
				deltas[j] = outputErrors[j] * SigmoidDerivative(outputs[j])
			}
		} else { // Calculate the delta for hidden layers
			nextLayer := nn.layers[i+1]
			for j := range layer.neurons {
				var error float64
				for k := range nextLayer.neurons {
					error += nextLayer.neurons[k].Weights[j] * deltas[k]
				}
				deltas[j] = error * SigmoidDerivative(outputs[j])
			}
		}

		// Update Weights and biases
		for j := range layer.neurons {
			neuron := &layer.neurons[j]
			for k := range neuron.Weights {
				neuron.Weights[k] += learnRate * deltas[j] * inputs[k]
			}
			neuron.Bias += learnRate * deltas[j]
		}
	}
}

// Initialize the network with the number of neurons in each layer
func NewNeuralNetwork(structure []int) *NeuralNetwork {
	rand.New(rand.NewSource(99))

	nn := &NeuralNetwork{layers: make([]Layer, len(structure)-1)}
	for i := 0; i < len(structure)-1; i++ {
		layer := Layer{neurons: make([]Neuron, structure[i+1])}
		for j := range layer.neurons {
			neuron := &layer.neurons[j]
			neuron.Weights = make([]float64, structure[i])
			for k := range neuron.Weights {
				neuron.Weights[k] = rand.Float64()*2 - 1 // Initialize Weights
			}
			neuron.Bias = rand.Float64()*2 - 1 // Initialize Bias
		}
		nn.layers[i] = layer
	}
	return nn
}
