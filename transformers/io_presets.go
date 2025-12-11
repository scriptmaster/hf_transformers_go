package transformers

import (
	"fmt"

	onnx "github.com/yalue/onnxruntime_go"
)

// IOPreset describes how we intend to wire inputs/outputs for a model.
type IOPreset int

const (
	// Automatic – fall back to GetInput/OutputInfo on the session.
	IOPresetAuto IOPreset = iota

	// Simple causal LM: [input_ids, attention_mask] -> [logits]
	IOPresetSimpleCausal

	// LFM2-style: input_ids, attention_mask, position_ids, past_* -> logits, present_*
	IOPresetLFM2
)

// resolveIONames sets m.inputNames and m.outputNames based on m.ioPreset.
// For known presets, it uses a static mapping.
// For IOPresetAuto (or unknown), it falls back to ONNX model introspection.
func (m *ModelForCausalLM) resolveIONames(onnxPath string) error {
	switch m.ioPreset {
	case IOPresetSimpleCausal:
		in, out, err := simpleCausalIONames()
		if err != nil {
			return err
		}
		m.inputNames = in
		m.outputNames = out
		return nil

	case IOPresetLFM2:
		in, out, err := lfm2IONames(m.config)
		if err != nil {
			return err
		}
		m.inputNames = in
		m.outputNames = out
		return nil

	case IOPresetAuto:
		fallthrough
	default:
		in, out, err := discoverIONamesFromModel(onnxPath)
		if err != nil {
			return err
		}
		m.inputNames = in
		m.outputNames = out
		return nil
	}
}

// simpleCausalIONames returns the standard GPT-style wiring:
// inputs:  input_ids, attention_mask
// outputs: logits
func simpleCausalIONames() ([]string, []string, error) {
	inputs := []string{"input_ids", "attention_mask"}
	outputs := []string{"logits"}
	return inputs, outputs, nil
}

// lfm2IONames builds the expected names for LFM2-350M-ONNX based on config.
// It does NOT inspect the graph; we rely on the naming convention exported
// by the original HF conversion script.
func lfm2IONames(cfg *Config) ([]string, []string, error) {
	if cfg == nil {
		return nil, nil, fmt.Errorf("lfm2IONames: config is nil")
	}

	inputs := []string{"input_ids", "attention_mask", "position_ids"}

	for layerIdx, t := range cfg.LayerTypes() {
		switch t {
		case "full_attention":
			inputs = append(inputs,
				fmt.Sprintf("past_key_values.%d.key", layerIdx),
				fmt.Sprintf("past_key_values.%d.value", layerIdx),
			)
		case "conv":
			inputs = append(inputs,
				fmt.Sprintf("past_conv.%d", layerIdx),
			)
		default:
			return nil, nil, fmt.Errorf("lfm2IONames: unsupported layer type %q", t)
		}
	}

	outputs := []string{"logits"}
	for _, name := range inputs {
		switch name {
		case "input_ids", "attention_mask", "position_ids":
			continue
		default:
			outputs = append(outputs, "present."+name)
		}
	}

	return inputs, outputs, nil
}

// discoverIONamesFromModel introspects the ONNX model to get input/output names.
// This is the fallback when no preset is known or ioPreset == IOPresetAuto.
func discoverIONamesFromModel(onnxPath string) ([]string, []string, error) {
	if onnxPath == "" {
		return nil, nil, fmt.Errorf("discoverIONamesFromModel: onnxPath is empty")
	}

	inputInfos, outputInfos, err := onnx.GetInputOutputInfo(onnxPath)
	if err != nil {
		return nil, nil, fmt.Errorf("discoverIONamesFromModel: %w", err)
	}

	inputs := make([]string, 0, len(inputInfos))
	for _, info := range inputInfos {
		inputs = append(inputs, info.Name)
	}

	outputs := make([]string, 0, len(outputInfos))
	for _, info := range outputInfos {
		outputs = append(outputs, info.Name)
	}

	return inputs, outputs, nil
}
