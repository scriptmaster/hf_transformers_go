package transformers

import (
	"errors"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"

	onnx "github.com/yalue/onnxruntime_go"
)

// ModelForCausalLM is our ONNX-backed language model wrapper.
type ModelForCausalLM struct {
	modelID     string
	config      *Config
	session     *onnx.DynamicAdvancedSession
	ioPreset    IOPreset
	inputNames  []string
	outputNames []string
	dtype       string // "q4", "fp16", etc.
	inputInfo   map[string]onnx.InputOutputInfo
}

// autoModelForCausalLM is the HF-style static dispatcher:
//   model, err := AutoModelForCausalLM.FromPretrained(...)
type autoModelForCausalLM struct{}

var AutoModelForCausalLM autoModelForCausalLM

// FromPretrained constructs the model from HF Hub.
func (autoModelForCausalLM) FromPretrained(
	modelID string,
	config *Config,
	dtype string,   // "q4", "fp16", "" -> chooses filename
	ioPreset IOPreset,
) (*ModelForCausalLM, error) {
	if config == nil {
		return nil, errors.New("AutoModelForCausalLM.FromPretrained: config is nil")
	}

	// choose ONNX filename from dtype
	filename := "onnx/model.onnx"
	switch dtype {
	case "q4":
		filename = "onnx/model_q4.onnx"
	case "fp16":
		filename = "onnx/model_fp16.onnx"
	default:
		filename = "onnx/model.onnx"
	}

	onnxPath, err := HFHubDownload(modelID, filename)
	if err != nil {
		return nil, fmt.Errorf("download onnx model: %w", err)
	}

	loadedFiles := []string{onnxPath}
	// Download external data files if present (best effort).
	if strings.HasSuffix(filename, ".onnx") {
		if files, _ := HFHubEnsureOptionalFiles(modelID, []string{filename + "_data"}); files != nil {
			if p, ok := files[filename+"_data"]; ok {
				loadedFiles = append(loadedFiles, p)
			}
		}
	}

	// Environment should be initialized once per process.
	if err := onnx.InitializeEnvironment(onnx.WithLogLevelWarning()); err != nil {
		return nil, fmt.Errorf("InitializeEnvironment: %w", err)
	}

	// Introspect input/output info to aid in creating zeroed optional inputs.
	inInfos, _, err := onnx.GetInputOutputInfo(onnxPath)
	if err != nil {
		return nil, fmt.Errorf("GetInputOutputInfo: %w", err)
	}
	inputInfo := make(map[string]onnx.InputOutputInfo, len(inInfos))
	for _, info := range inInfos {
		inputInfo[info.Name] = info
	}

	m := &ModelForCausalLM{
		modelID:  modelID,
		config:   config,
		ioPreset: ioPreset,
		dtype:    dtype,
		inputInfo: inputInfo,
	}

	if err := m.resolveIONames(onnxPath); err != nil {
		return nil, err
	}

	sess, err := onnx.NewDynamicAdvancedSession(
		onnxPath,
		m.inputNames,
		m.outputNames,
		nil, // session options
	)
	if err != nil {
		return nil, fmt.Errorf("create ONNX session: %w", err)
	}

	m.session = sess

	logModelLoadInfo(modelID)

	return m, nil
}

// GenerationOptions describes generation parameters for a call.
type GenerationOptions struct {
	MaxNewTokens int
	DoSample     bool
	Streamer     func(ev PipelineStreamEvent) bool // return false to stop early
	StopSequences []string
}

// Generate runs a chat-style generation loop with optional streaming.
// It currently supports batch=1 only.
func (m *ModelForCausalLM) Generate(
	tokenizer *Tokenizer,
	inputIDs [][]int64,
	attentionMask [][]int64,
	opts GenerationOptions,
) ([][]int64, error) {
	if tokenizer == nil {
		return nil, errors.New("Generate: tokenizer is nil")
	}
	if m.session == nil {
		return nil, errors.New("Generate: session is nil")
	}
	if len(inputIDs) != 1 || len(attentionMask) != 1 {
		return nil, errors.New("Generate: only batch=1 is supported currently")
	}
	if opts.MaxNewTokens <= 0 {
		opts.MaxNewTokens = 128
	}

	switch m.ioPreset {
	case IOPresetSimpleCausal:
		return m.generateSimpleCausal(tokenizer, inputIDs[0], attentionMask[0], opts)
	case IOPresetLFM2:
		// Skeleton; can be extended to full LFM2 KV cache.
		return m.generateSimpleCausal(tokenizer, inputIDs[0], attentionMask[0], opts)
	case IOPresetAuto:
		fallthrough
	default:
		return m.generateSimpleCausal(tokenizer, inputIDs[0], attentionMask[0], opts)
	}
}

// generateSimpleCausal implements a simple greedy loop using only input_ids
// and attention_mask and reading logits.
func (m *ModelForCausalLM) generateSimpleCausal(
	tokenizer *Tokenizer,
	curIDs []int64,
	curMask []int64,
	opts GenerationOptions,
) ([][]int64, error) {
	var generated []int64
	eosID := m.config.EOS_TOKEN_ID()

	var fullText string

	for step := 0; step < opts.MaxNewTokens; step++ {
		// Prepare input tensors
		inputTensor, err := tensorFromInt64s(curIDs, []int64{1, int64(len(curIDs))})
		if err != nil {
			return nil, fmt.Errorf("create input_ids tensor: %w", err)
		}
		maskTensor, err := tensorFromInt64s(curMask, []int64{1, int64(len(curMask))})
		if err != nil {
			inputTensor.Destroy()
			return nil, fmt.Errorf("create attention_mask tensor: %w", err)
		}

		inputs := make([]onnx.Value, len(m.inputNames))
		var toDestroy []onnx.Value
		for i, name := range m.inputNames {
			switch name {
			case "input_ids":
				inputs[i] = inputTensor
			case "attention_mask":
				inputs[i] = maskTensor
			case "position_ids":
				pos := make([]int64, len(curIDs))
				for j := range pos {
					pos[j] = int64(j)
				}
				t, err := tensorFromInt64s(pos, []int64{1, int64(len(pos))})
				if err != nil {
					inputTensor.Destroy()
					maskTensor.Destroy()
					return nil, fmt.Errorf("create position_ids tensor: %w", err)
				}
				inputs[i] = t
				toDestroy = append(toDestroy, t)
			default:
				t, err := m.zeroTensorForInput(name, len(curIDs))
				if err != nil {
					inputTensor.Destroy()
					maskTensor.Destroy()
					for _, v := range toDestroy {
						v.Destroy()
					}
					return nil, err
				}
				inputs[i] = t
				toDestroy = append(toDestroy, t)
			}
		}

		outputs := make([]onnx.Value, len(m.outputNames))

		if err := m.session.Run(inputs, outputs); err != nil {
			inputTensor.Destroy()
			maskTensor.Destroy()
			for _, v := range toDestroy {
				v.Destroy()
			}
			return nil, fmt.Errorf("onnx Run: %w", err)
		}

		inputTensor.Destroy()
		maskTensor.Destroy()
		for _, v := range toDestroy {
			v.Destroy()
		}

		var logitsTensor *onnx.Tensor[float32]
		for i, name := range m.outputNames {
			if name != "logits" {
				if outputs[i] != nil {
					// Clean up any auto-allocated outputs we don't consume.
					_ = outputs[i].Destroy()
				}
				continue
			}

			val := outputs[i]
			if val == nil {
				return nil, errors.New("onnx output 'logits' missing")
			}

			t, ok := val.(*onnx.Tensor[float32])
			if !ok {
				return nil, errors.New("onnx 'logits' is not a float32 Tensor")
			}
			logitsTensor = t
		}

		if logitsTensor == nil {
			return nil, errors.New("onnx output 'logits' missing")
		}
		raw := logitsTensor.GetData()
		shape := logitsTensor.GetShape()
		if len(shape) != 3 {
			return nil, fmt.Errorf("unexpected logits shape: %v", shape)
		}
		vocabSize := int(shape[2])

		start := (len(curIDs) - 1) * vocabSize
		end := start + vocabSize
		lastLogits := raw[start:end]

		// For now: greedy. You can add sampling using softmaxF32/sampleFromProbsF32.
		nextID := int64(argmaxF32(lastLogits))
		logitsTensor.Destroy()

		generated = append(generated, nextID)
		curIDs = append(curIDs, nextID)
		curMask = append(curMask, 1)

		deltaText := ""
		if tokenizer != nil {
			txt, err := tokenizer.Decode([]int64{nextID})
			if err == nil {
				deltaText = txt
				fullText += deltaText
			}
		}

		// Stop sequence handling (string-based).
		stopHit := false
		for _, stop := range opts.StopSequences {
			if stop == "" {
				continue
			}
			if idx := strings.Index(fullText, stop); idx >= 0 {
				fullText = fullText[:idx]
				deltaText = "" // avoid streaming the stop tail
				stopHit = true
				break
			}
		}

		done := eosID >= 0 && nextID == eosID

		if opts.Streamer != nil {
			ev := PipelineStreamEvent{
				TokenID:   nextID,
				DeltaText: deltaText,
				FullText:  fullText,
				Step:      step,
				Done:      done || stopHit,
			}
			if !opts.Streamer(ev) {
				break
			}
		}

		if done || stopHit {
			break
		}
	}

	return [][]int64{generated}, nil
}

func logModelLoadInfo(modelID string) {
	files := listDownloaded(modelID)
	rssMB := currentRSSMB()
	log.Printf("model loaded: repo=%s files=%v rss_mb=%.1f gpu_mb=0", modelID, files, rssMB)
}

func currentRSSMB() float64 {
	data, err := os.ReadFile("/proc/self/statm")
	if err != nil {
		return 0
	}
	fields := strings.Fields(string(data))
	if len(fields) < 2 {
		return 0
	}
	residentPages, err := strconv.ParseInt(fields[1], 10, 64)
	if err != nil {
		return 0
	}
	pageSize := int64(os.Getpagesize())
	return float64(residentPages*pageSize) / (1024.0 * 1024.0)
}

func (m *ModelForCausalLM) zeroTensorForInput(name string, seqLen int) (onnx.Value, error) {
	info, ok := m.inputInfo[name]
	if !ok {
		return nil, fmt.Errorf("Generate: unsupported input name %q", name)
	}
	isCache := strings.Contains(name, "past") || strings.Contains(name, "cache")
	shape := make([]int64, len(info.Dimensions))
	for i, d := range info.Dimensions {
		if d <= 0 {
			if i == 0 {
				shape[i] = 1 // batch dim must be >=1
			} else if isCache {
				shape[i] = 0 // allow empty cache length
			} else {
				shape[i] = 1
			}
			// For non-cache, try to use seqLen if dimension is undefined and position-like
			if !isCache && i == len(info.Dimensions)-1 && seqLen > 0 {
				shape[i] = int64(seqLen)
			}
		} else {
			shape[i] = d
		}
	}

	switch info.DataType {
	case onnx.TensorElementDataTypeInt64:
		count := int64(1)
		for _, d := range shape {
			count *= d
		}
		data := make([]int64, count)
		return tensorFromInt64s(data, shape)
	default:
		count := int64(1)
		for _, d := range shape {
			count *= d
		}
		data := make([]float32, count)
		return tensorFromFloat32s(data, shape)
	}
}
