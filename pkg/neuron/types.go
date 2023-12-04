package neuron

// Neuron 结构体
type Neuron struct {
	Weights []float64 `yaml:"weights" toml:"weights" json:"weights"`
	Bias    float64   `yaml:"bias" toml:"bias" json:"bias"`
	Delta   float64   `yaml:"delta" toml:"delta" json:"delta"`
	Output  float64   `yaml:"output" toml:"output" json:"output"`
}
