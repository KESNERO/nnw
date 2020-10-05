package nnw

import (
	"bytes"
	"encoding/json"
	"io"
	"io/ioutil"
	"math"
	"math/rand"
	"os"
)

func Gauss() float64 {
	μ := 0.0
	σ := 0.25
	x := rand.Float64()
	result := 1 / (math.Sqrt(2*math.Pi) * σ) * math.Pow(math.E, -math.Pow(x-μ, 2)/(2*math.Pow(σ, 2)))
	return result
}

type Network struct {
	LayerDefine   []int
	BatchSize     int
	LearningRate  float64
	Layers        []*Layer
	SpecialLayers []*Layer
	W             [][][]float64 // W[0][1][2] means from first layer's second position
	// to next layer's third position
}

func NewNetwork(ld []int, lr float64) *Network {
	network := new(Network)
	network.LayerDefine = ld
	network.LearningRate = lr
	network.Layers = make([]*Layer, 0)
	network.W = make([][][]float64, 0)
	for i, ln := range ld {
		network.Layers = append(network.Layers, NewLayer(ln, "normal"))
		network.SpecialLayers = append(network.SpecialLayers, NewLayer(ln, "normal"))
		if i > 0 {
			w := make([][]float64, ld[i-1])
			for j := range w {
				w[j] = make([]float64, ln)
				for k := range w[j] {
					w[j][k] = Gauss()
				}
			}
			network.W = append(network.W, w)
			if i < len(ld)-1 {
				network.Layers = append(network.Layers, NewLayer(ln, "activation"))
			}
		}
	}
	return network
}

func (network *Network) ForwardSpread(input [][]float64) [][]float64 {
	for i := range network.SpecialLayers {
		network.SpecialLayers[i].Reset()
	}
	var output = make([][]float64, len(input))
	for i := range input {
		curLayerIndex := 0
		curSpecialLayerIndex := 0
		curWIndex := 0
		network.Layers[curLayerIndex].Input(input[i])
		network.SpecialLayers[curLayerIndex].Input(input[i])
		for curLayerIndex < len(network.Layers)-1 {
			switch network.Layers[curLayerIndex].Type {
			case "normal":
				network.Layers[curLayerIndex+1].Input(network.Layers[curLayerIndex].RightProduct(network.W[curWIndex]))
				network.SpecialLayers[curSpecialLayerIndex+1].Plus(network.Layers[curLayerIndex+1].Output())
				curSpecialLayerIndex++
			case "activation":
				if curLayerIndex == len(network.Layers)-2 {
					network.Layers[curLayerIndex+1].Input(network.Layers[curLayerIndex].Activate("Sigmoid"))
				} else {
					network.Layers[curLayerIndex+1].Input(network.Layers[curLayerIndex].Activate("LeRU"))
				}
			}
			curLayerIndex++
		}
		output[i] = make([]float64, network.Layers[curLayerIndex].Size)
		copy(output[i], network.Layers[curLayerIndex].Neural)
	}
	for i := range network.SpecialLayers {
		network.SpecialLayers[i].Divide(float64(len(input)))
	}
	return output
}

func (network *Network) UpdateW(curW [][]float64, left *Layer, err []float64) {
	for i := range curW {
		for j := range curW[i] {
			curW[i][j] -= network.LearningRate * left.Neural[i] * err[j]
		}
	}
}

func (network *Network) PreprocessSigmoid(err []float64, l *Layer) {
	for i := range err {
		err[i] *= l.Neural[i] * (l.Neural[i-1])
	}
}

func (network *Network) BackPropagation(err []float64) {
	curSpecialLayerIndex := len(network.SpecialLayers)-1
	curWIndex := len(network.W)-1
	var curErr = make([]float64, len(err))
	copy(curErr, err)
	network.PreprocessSigmoid(curErr, network.SpecialLayers[curSpecialLayerIndex])
	network.SpecialLayers[curSpecialLayerIndex].Input(curErr)
	for curSpecialLayerIndex > 0 {
		nextErr := network.SpecialLayers[curSpecialLayerIndex].LeftProduct(network.W[curWIndex])
		network.UpdateW(network.W[curWIndex], network.SpecialLayers[curSpecialLayerIndex-1], curErr)
		curErr = make([]float64, len(nextErr))
		copy(curErr, nextErr)
		curSpecialLayerIndex--
	}
}

func (network *Network) SaveW() {
	f, err := os.OpenFile("w.csv", os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0644)
	if err != nil {
		panic(err)
	}
	stream, err := json.Marshal(network.W)
	if err != nil {
		panic(err)
	}
	_, _ = io.Copy(f, bytes.NewReader(stream))
}

func (network *Network) LoadW() {
	f, err := os.OpenFile("w.csv", os.O_RDONLY, 0644)
	if err != nil {
		panic(err)
	}
	stream, err := ioutil.ReadAll(f)
	if err != nil {
		panic(err)
	}
	if err = json.Unmarshal(stream, network.W); err != nil {
		panic(err)
	}
}
