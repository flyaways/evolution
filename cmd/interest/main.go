package main

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
)

type Neuron struct {
	ID       int
	Value    float64
	Synapses []*Synapse
}

type Synapse struct {
	Weight float64
	Input  *Neuron
	Output *Neuron
}

type NeuralNetwork struct {
	Neurons []*Neuron
}

func NewNeuron(id int) *Neuron {
	return &Neuron{
		ID:       id,
		Value:    0.0,
		Synapses: make([]*Synapse, 0),
	}
}

func (n *Neuron) AddSynapse(weight float64, output *Neuron) {
	synapse := &Synapse{
		Weight: weight,
		Input:  n,
		Output: output,
	}
	n.Synapses = append(n.Synapses, synapse)
}

func (n *Neuron) Activate() {
	sum := 0.0
	for _, synapse := range n.Synapses {
		sum += synapse.Weight * synapse.Input.Value
	}
	n.Value = sigmoid(sum)
}

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func (nn *NeuralNetwork) ForwardPass(input float64) float64 {
	nn.Neurons[0].Value = input
	for _, neuron := range nn.Neurons {
		neuron.Activate()
	}
	return nn.Neurons[len(nn.Neurons)-1].Value
}

func (nn *NeuralNetwork) Backpropagate(target float64, learningRate float64) {
	// A very basic and naive backpropagation approach for demonstration purposes
	for _, neuron := range nn.Neurons {
		for _, synapse := range neuron.Synapses {
			predictionError := target - synapse.Output.Value
			curiosityReward := math.Abs(predictionError) // The reward for curiosity is the prediction error
			synapse.Weight += learningRate * curiosityReward * synapse.Input.Value
		}
	}
}

func (nn *NeuralNetwork) Save(filename string) error {
	data, err := json.Marshal(nn)
	if err != nil {
		return err
	}
	return os.WriteFile(filename, data, 0644)
}

func (nn *NeuralNetwork) Load(filename string) error {
	data, err := os.ReadFile(filename)
	if err != nil {
		return err
	}
	return json.Unmarshal(data, nn)
}

func main() {
	rand.New(rand.NewSource(99))

	// Create a simple neural network with 3 neurons connected in series
	nn := NeuralNetwork{}

	n1 := NewNeuron(1)
	n2 := NewNeuron(2)
	n3 := NewNeuron(3)

	n1.AddSynapse(rand.NormFloat64(), n2)
	n2.AddSynapse(rand.NormFloat64(), n3)

	nn.Neurons = append(nn.Neurons, n1, n2, n3)

	// Train the network with a curiosity-driven approach
	for i := 0; i < 1000; i++ {
		input := rand.Float64()*2 - 1 // Random input between -1 and 1
		target := math.Sin(input)     // A target function that the network doesn't know

		output := nn.ForwardPass(input)
		nn.Backpropagate(target, 0.01)

		fmt.Printf("Epoch %d - Input: %.2f, Predicted: %.2f, Target: %.2f\n", i, input, output, target)
	}

	// Save the neural network state to a file
	if err := nn.Save("neural_network.json"); err != nil {
		fmt.Println("Error saving neural network:", err)
	}

	// Load the neural network state from a file
	if err := nn.Load("neural_network.json"); err != nil {
		fmt.Println("Error loading neural network:", err)
	}
}
