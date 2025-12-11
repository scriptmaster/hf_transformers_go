package transformers

import (
	"math"

	onnx "github.com/yalue/onnxruntime_go"
)

// tensorFromInt64s wraps []int64 into an ONNX tensor with the given shape.
func tensorFromInt64s(data []int64, shape []int64) (*onnx.Tensor[int64], error) {
	sh := onnx.NewShape(shape...)
	return onnx.NewTensor(sh, data)
}

// tensorFromFloat32s wraps []float32 into an ONNX tensor with the given shape.
func tensorFromFloat32s(data []float32, shape []int64) (*onnx.Tensor[float32], error) {
	sh := onnx.NewShape(shape...)
	return onnx.NewTensor(sh, data)
}

// argmaxF32 returns the index of the largest value in xs.
// If xs is empty, returns 0.
func argmaxF32(xs []float32) int {
	if len(xs) == 0 {
		return 0
	}
	maxIdx := 0
	maxVal := xs[0]
	for i := 1; i < len(xs); i++ {
		if xs[i] > maxVal {
			maxVal = xs[i]
			maxIdx = i
		}
	}
	return maxIdx
}

// softmaxF32 converts logits in-place to probabilities.
func softmaxF32(xs []float32) {
	if len(xs) == 0 {
		return
	}

	maxVal := xs[0]
	for _, v := range xs[1:] {
		if v > maxVal {
			maxVal = v
		}
	}

	sum := float32(0)
	for i, v := range xs {
		e := float32(math.Exp(float64(v - maxVal)))
		xs[i] = e
		sum += e
	}
	if sum == 0 {
		return
	}
	inv := 1 / sum
	for i := range xs {
		xs[i] *= inv
	}
}

// sampleFromProbsF32 samples an index from a probability distribution xs.
// Assumes xs are normalized to sum ~1 (softmaxF32 can be used first).
func sampleFromProbsF32(xs []float32, rnd func() float32) int {
	r := rnd()
	acc := float32(0)
	for i, p := range xs {
		acc += p
		if r < acc {
			return i
		}
	}
	if len(xs) == 0 {
		return 0
	}
	return len(xs) - 1
}
