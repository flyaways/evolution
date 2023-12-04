package neuron

// Train 单个神经元的训练函数
func (n *Neuron) Train(inputs []float64, expected float64, learnRate float64) {
	// 前向传播计算输出
	Output := n.FeedForward(inputs)

	// 计算误差
	error := expected - Output

	// 计算Delta（用于权重更新）
	n.Delta = error * SigmoidDerivative(Output)

	// 更新权重和偏置
	for i := range n.Weights {
		n.Weights[i] += n.Delta * inputs[i] * learnRate
	}
	n.Bias += n.Delta * learnRate
}
