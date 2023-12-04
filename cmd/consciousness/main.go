package main

import (
	"fmt"

	"github.com/flyaways/evolution/pkg/nn"
)

func main() {
	// XOR problem
	inputs := [][]float64{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	}
	expected := [][]float64{
		{0},
		{1},
		{1},
		{0},
	}

	// Create a neural network with 2 inputs, two hidden layers of 3 neurons each, and 1 output
	nn := nn.NewNeuralNetwork([]int{
		2, //inputs layers
		3, //hidden layers
		3, //hidden layers
		1, //output layers
	})

	// Train the network
	learnRate := 0.5
	for epoch := 0; epoch < 10000; epoch++ {
		for i, input := range inputs {
			nn.BackPropagate(
				input,
				expected[i],
				learnRate,
			)
		}
	}

	// Test the network
	for _, input := range inputs {
		output := nn.Forward(input)
		fmt.Printf("Input: %v, Output: %v\n", input, output)
	}
}
