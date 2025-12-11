package transformers

import (
	"fmt"
	"strings"
)

// Pipeline is the exported HF-style entry point:
//
//	generator, err := Pipeline("text-generation", modelID, map[string]any{"dtype": "q4"})
//
// Internally it delegates to the lowercase pipelineImpl, so you can define
// a small-p alias in your own code if you dot-import the package:
//
//	var pipeline = transformers.Pipeline
func Pipeline(
	task string,
	modelID string,
	options map[string]any,
) (Generator, error) {
	return pipelineImpl(task, modelID, options)
}

// pipelineImpl is the internal implementation with a smaller name.
// It mirrors the JS/Python `pipeline(...)` signature conceptually.
func pipelineImpl(
	task string,
	modelID string,
	options map[string]any,
) (Generator, error) {
	if task != "text-generation" {
		return nil, fmt.Errorf("pipeline: task %q not implemented", task)
	}

	if options == nil {
		options = map[string]any{}
	}
	dtype, _ := options["dtype"].(string)
	if dtype == "" {
		dtype = "q4"
	}

	// 1. Config
	config, err := AutoConfig.FromPretrained(modelID)
	if err != nil {
		return nil, fmt.Errorf("load config: %w", err)
	}

	// 2. Tokenizer
	tokenizer, err := AutoTokenizer.FromPretrained(modelID)
	if err != nil {
		return nil, fmt.Errorf("load tokenizer: %w", err)
	}

	// Prefer auto IO discovery to match model-defined inputs/outputs.
	ioPreset := IOPresetAuto

	// 3. Model
	model, err := AutoModelForCausalLM.FromPretrained(
		modelID,
		config,
		dtype,
		ioPreset,
	)
	if err != nil {
		return nil, fmt.Errorf("load model: %w", err)
	}

	// 4. Closure = generator(messages, options)
	generator := func(
		messages []ChatMessage,
		callOptions map[string]any,
	) ([]map[string]any, error) {
		if callOptions == nil {
			callOptions = map[string]any{}
		}

		// Default to a short cap to avoid run-on generations.
		maxNewTokens := 32
		if v, ok := callOptions["max_new_tokens"]; ok {
			switch t := v.(type) {
			case int:
				maxNewTokens = t
			case float64:
				maxNewTokens = int(t)
			}
		}
		doSample := false
		if v, ok := callOptions["do_sample"]; ok {
			if b, ok := v.(bool); ok {
				doSample = b
			}
		}

		var streamerFn func(PipelineStreamEvent) bool
		if v, ok := callOptions["streamer"]; ok {
			if fn, ok := v.(func(PipelineStreamEvent) bool); ok {
				streamerFn = fn
			}
		}

		// 4a. Encode chat
		inputIDsBatch, attnBatch, _, _, err := tokenizer.EncodeChat(messages)
		if err != nil {
			return nil, fmt.Errorf("EncodeChat: %w", err)
		}

		// 4b. Generate token IDs
		stopSeqs := parseStopSequences(callOptions["stop"])
		if len(stopSeqs) == 0 && len(config.StopStrings()) > 0 {
			stopSeqs = config.StopStrings()
		}
		if len(stopSeqs) == 0 {
			stopSeqs = []string{"\nUser:", "\nuser:", "\nAssistant:", "\nassistant:"}
		}

		genOpts := GenerationOptions{
			MaxNewTokens:  maxNewTokens,
			DoSample:      doSample,
			Streamer:      streamerFn,
			StopSequences: stopSeqs,
		}
		generatedBatch, err := model.Generate(tokenizer, inputIDsBatch, attnBatch, genOpts)
		if err != nil {
			return nil, fmt.Errorf("Generate: %w", err)
		}

		// 4c. Decode generated tokens to text
		texts, err := tokenizer.BatchDecode(generatedBatch)
		if err != nil {
			return nil, fmt.Errorf("BatchDecode: %w", err)
		}
		for i, txt := range texts {
			texts[i] = truncateAtStops(txt, stopSeqs)
		}

		// Wrap into HF-style output:
		// [{ "generated_text": [ { "role": "assistant", "content": text } ] }]
		out := make([]map[string]any, len(texts))
		for i, txt := range texts {
			trimmed := strings.TrimSpace(txt)
			out[i] = map[string]any{
				"generated_text": []map[string]any{
					{
						"role":    "assistant",
						"content": trimmed,
					},
				},
			}
		}
		return out, nil
	}

	return generator, nil
}

func parseStopSequences(v any) []string {
	switch t := v.(type) {
	case nil:
		return nil
	case string:
		if t == "" {
			return nil
		}
		return []string{t}
	case []string:
		return t
	case []any:
		var out []string
		for _, x := range t {
			if s, ok := x.(string); ok && s != "" {
				out = append(out, s)
			}
		}
		return out
	default:
		return nil
	}
}

func truncateAtStops(s string, stops []string) string {
	out := s
	for _, stop := range stops {
		if stop == "" {
			continue
		}
		if idx := strings.Index(out, stop); idx >= 0 {
			out = out[:idx]
		}
	}
	return strings.TrimSpace(out)
}
