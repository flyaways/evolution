package main

import (
	"fmt"
	"math/rand"

	"github.com/flyaways/evolution/pkg/neuron"
)

// 要实现一个更完整的神经元工作原理模拟，我们需要考虑以下几个方面：

// 神经元模型：包括输入、权重、偏置和激活函数。
// 前向传播：神经元的输出是输入、权重和偏置的加权和，通过激活函数处理。
// 反向传播（可选）：一种学习算法，用于调整权重和偏置以最小化误差。
// 以下是Go语言中一个简单的神经元模型的实现，包括前向传播和反向传播：

// `Neuron` 结构体，它包含了权重、偏置、输出值和用于反向传播的 delta 值。我们在 `NewNeuron` 函数中初始化了权重和偏置。
// 在 `FeedForward` 方法中，我们实现了前向传播的逻辑，计算了加权输入和偏置的总和，然后通过 Sigmoid 激活函数。
// 在 `Train` 方法中，我们实现了一个简单的反向传播算法。
// 首先，我们计算了神经元输出与期望输出之间的误差，然后计算了 delta（误差的梯度），这将用于更新权重和偏置。
// 学习率 `learnRate` 决定了在每次迭代中我们调整权重和偏置的程度。
// 在 `main` 函数中，我们创建了一个神经元，并用一组训练数据来训练它。
// 这个例子中，我们使用了 XOR 问题的数据集，XOR 是一个典型的非线性问题，不能由单个神经元解决，但这里我们用它来示范训练过程。
// 训练完成后，我们遍历训练数据，使用训练好的神经元来预测输出，并打印结果。
// 请注意，这个例子中的单个神经元无法准确解决 XOR 问题，因为 XOR 问题本质上是非线性的，而单个神经元只能解决线性可分问题。
// 要解决 XOR 问题，你需要一个包含至少一个隐藏层的神经网络。
// 但是，这个代码示例提供了如何在 Go 中实现一个神经元的基本框架，包括它的前向传播和简单的反向传播训练过程。

func main() {
	rand.New(rand.NewSource(99))

	// 创建神经元
	neuron := neuron.NewNeuron(2)

	// 训练数据
	trainingInputs := [][]float64{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	}
	trainingOutputs := []float64{0, 1, 1, 0} // XOR 问题的输出

	// 训练过程
	learnRate := 0.1
	for epoch := 0; epoch < 10000; epoch++ {
		for i, inputs := range trainingInputs {
			neuron.Train(inputs, trainingOutputs[i], learnRate)
		}
	}

	// 测试训练结果
	for _, inputs := range trainingInputs {
		fmt.Printf("Inputs: %v, Output: %f\n", inputs, neuron.FeedForward(inputs))
	}
}
