package transformers

import (
	"encoding/json"
	"fmt"
	"os"
)

// Config holds model configuration loaded from config.json.
type Config struct {
	modelType         string
	vocabSize         int
	eosTokenID        int64
	bosTokenID        int64
	padTokenID        int64
	numHiddenLayers   int
	numAttentionHeads int
	numKeyValueHeads  int
	hiddenSize        int
	convLCache        int
	layerTypes        []string

	raw map[string]any

	// generation config (optional)
	stopStrings []string
}

// AutoConfig is the HF-style static dispatcher:
//   config, err := AutoConfig.FromPretrained(modelID)
type autoConfig struct{}

var AutoConfig autoConfig

// FromPretrained loads config.json from HF Hub for the given model ID.
func (autoConfig) FromPretrained(
	modelID string,
) (*Config, error) {
	cfgPath, err := HFHubDownload(modelID, "config.json")
	if err != nil {
		return nil, err
	}
	cfgData, err := os.ReadFile(cfgPath)
	if err != nil {
		return nil, err
	}
	var raw map[string]any
	if err := json.Unmarshal(cfgData, &raw); err != nil {
		return nil, err
	}

	getInt := func(key string, def int) int {
		if v, ok := raw[key]; ok {
			switch t := v.(type) {
			case float64:
				return int(t)
			case int:
				return t
			}
		}
		return def
	}
	getInt64 := func(key string, def int64) int64 {
		if v, ok := raw[key]; ok {
			switch t := v.(type) {
			case float64:
				return int64(t)
			case int64:
				return t
			case int:
				return int64(t)
			}
		}
		return def
	}
	getString := func(key string) string {
		if v, ok := raw[key]; ok {
			if s, ok := v.(string); ok {
				return s
			}
		}
		return ""
	}

	cfg := &Config{
		modelType:         getString("model_type"),
		vocabSize:         getInt("vocab_size", 0),
		eosTokenID:        getInt64("eos_token_id", -1),
		bosTokenID:        getInt64("bos_token_id", -1),
		padTokenID:        getInt64("pad_token_id", -1),
		numHiddenLayers:   getInt("num_hidden_layers", 0),
		numAttentionHeads: getInt("num_attention_heads", 0),
		numKeyValueHeads:  getInt("num_key_value_heads", 0),
		hiddenSize:        getInt("hidden_size", 0),
		convLCache:        getInt("conv_l_cache", 0),
		raw:               raw,
	}

	if lt, ok := raw["layer_types"].([]any); ok {
		cfg.layerTypes = make([]string, len(lt))
		for i, v := range lt {
			if s, ok := v.(string); ok {
				cfg.layerTypes[i] = s
			}
		}
	}

	if cfg.modelType == "" {
		return nil, fmt.Errorf("AutoConfig: model_type missing in config.json")
	}

	// Merge generation_config.json if present (best effort).
	cfg.applyGenerationConfig(modelID)

	return cfg, nil
}

// accessors
func (c *Config) ModelType() string        { return c.modelType }
func (c *Config) VocabSize() int           { return c.vocabSize }
func (c *Config) EOS_TOKEN_ID() int64      { return c.eosTokenID }
func (c *Config) BOS_TOKEN_ID() int64      { return c.bosTokenID }
func (c *Config) PAD_TOKEN_ID() int64      { return c.padTokenID }
func (c *Config) NumHiddenLayers() int     { return c.numHiddenLayers }
func (c *Config) NumAttentionHeads() int   { return c.numAttentionHeads }
func (c *Config) NumKeyValueHeads() int    { return c.numKeyValueHeads }
func (c *Config) HiddenSize() int          { return c.hiddenSize }
func (c *Config) ConvLCache() int          { return c.convLCache }
func (c *Config) LayerTypes() []string     { return c.layerTypes }
func (c *Config) Raw() map[string]any      { return c.raw }
func (c *Config) StopStrings() []string    { return c.stopStrings }

func (c *Config) applyGenerationConfig(modelID string) {
	genPath, err := HFHubDownload(modelID, "generation_config.json")
	if err != nil {
		return
	}
	data, err := os.ReadFile(genPath)
	if err != nil {
		return
	}
	var gen map[string]any
	if err := json.Unmarshal(data, &gen); err != nil {
		return
	}
	// Override token IDs if present
	if v, ok := gen["eos_token_id"]; ok {
		if id, ok2 := toInt64(v); ok2 {
			c.eosTokenID = id
		}
	}
	if v, ok := gen["bos_token_id"]; ok {
		if id, ok2 := toInt64(v); ok2 {
			c.bosTokenID = id
		}
	}
	if v, ok := gen["pad_token_id"]; ok {
		if id, ok2 := toInt64(v); ok2 {
			c.padTokenID = id
		}
	}
	// Collect stop strings if present
	if v, ok := gen["stop"]; ok {
		switch t := v.(type) {
		case string:
			if t != "" {
				c.stopStrings = []string{t}
			}
		case []any:
			for _, x := range t {
				if s, ok := x.(string); ok && s != "" {
					c.stopStrings = append(c.stopStrings, s)
				}
			}
		}
	}
}

func toInt64(v any) (int64, bool) {
	switch t := v.(type) {
	case float64:
		return int64(t), true
	case int:
		return int64(t), true
	case int64:
		return t, true
	}
	return 0, false
}
