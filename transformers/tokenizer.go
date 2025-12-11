package transformers

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/sugarme/tokenizer"
	"github.com/sugarme/tokenizer/pretrained"
)

// Tokenizer wraps sugarme/tokenizer with a HF-like interface.
type Tokenizer struct {
	tok *tokenizer.Tokenizer
}

// AutoTokenizer is the HF-style static dispatcher:
//
//	tokenizer, err := AutoTokenizer.FromPretrained(modelID)
type autoTokenizer struct{}

var AutoTokenizer autoTokenizer

// FromPretrained loads tokenizer.json from HF Hub.
func (autoTokenizer) FromPretrained(
	modelID string,
) (*Tokenizer, error) {
	tokenizerPath, err := HFHubDownload(modelID, "tokenizer.json")
	if err != nil {
		return nil, err
	}

	// Best-effort fetch of auxiliary tokenizer assets; missing files are skipped.
	_, _ = HFHubEnsureOptionalFiles(modelID, modelFilesList([]string{
		"tokenizer_config.json",
		"special_tokens_map.json",
		"vocab.json",
		"merges.txt",
	}))

	sanitizedPath, err := sanitizeTokenizerJSON(tokenizerPath)
	if err != nil {
		return nil, err
	}

	tok, err := pretrained.FromFile(sanitizedPath)
	if err != nil {
		return nil, fmt.Errorf("AutoTokenizer: %w", err)
	}

	return &Tokenizer{tok: tok}, nil
}

// Encode plain text into IDs.
func (t *Tokenizer) Encode(text string, addSpecialTokens bool) ([]int64, error) {
	enc, err := t.tok.EncodeSingle(text, addSpecialTokens)
	if err != nil {
		return nil, err
	}
	ids := enc.Ids
	out := make([]int64, len(ids))
	for i, v := range ids {
		out[i] = int64(v)
	}
	return out, nil
}

// Decode IDs into plain text.
func (t *Tokenizer) Decode(ids []int64) (string, error) {
	uids := make([]int, len(ids))
	for i, v := range ids {
		uids[i] = int(v)
	}
	return t.tok.Decode(uids, true), nil
}

// BatchDecode helper.
func (t *Tokenizer) BatchDecode(batch [][]int64) ([]string, error) {
	res := make([]string, len(batch))
	for i, seq := range batch {
		txt, err := t.Decode(seq)
		if err != nil {
			return nil, err
		}
		res[i] = txt
	}
	return res, nil
}

// Very minimal chat template (v1):
// - system lines first (if any)
// - each user/assistant line prefixed with "User:" / "Assistant:"
// - always end with "Assistant:" to cue the model to answer next.
// This is a fallback when no chat_template is provided by the model config.
func (t *Tokenizer) renderChatTemplate(messages []ChatMessage) (string, error) {
	var b strings.Builder
	// collect system text first
	for _, m := range messages {
		if m.Role == RoleSystem {
			b.WriteString("System: ")
			b.WriteString(m.Content)
			b.WriteString("\n")
		}
	}
	// then user/assistant turns in order
	for _, m := range messages {
		if m.Role == RoleSystem {
			continue
		}
		role := "User"
		if m.Role == RoleAssistant {
			role = "Assistant"
		}
		b.WriteString(role)
		b.WriteString(": ")
		b.WriteString(m.Content)
		b.WriteString("\n")
	}
	return b.String(), nil
}

// EncodeChat encodes the full chat into input IDs and attention mask.
func (t *Tokenizer) EncodeChat(
	messages []ChatMessage,
) (inputIDs [][]int64, attentionMask [][]int64, promptLen int, rawText string, err error) {
	rawText, err = t.renderChatTemplate(messages)
	if err != nil {
		return nil, nil, 0, "", err
	}
	ids, err := t.Encode(rawText, true)
	if err != nil {
		return nil, nil, 0, "", err
	}
	attn := make([]int64, len(ids))
	for i := range attn {
		attn[i] = 1
	}
	return [][]int64{ids}, [][]int64{attn}, len(ids), rawText, nil
}

func (t *Tokenizer) Info() string {
	return fmt.Sprintf("Tokenizer(vocab=%d)", t.tok.GetVocabSize(true))
}

// sanitizeTokenizerJSON rewrites unsupported regex patterns (like negative
// lookaheads) into Go-regexp-compatible forms and returns a path to the
// sanitized copy.
func sanitizeTokenizerJSON(origPath string) (string, error) {
	raw, err := os.ReadFile(origPath)
	if err != nil {
		return "", err
	}

	// Replace unsupported negative lookahead with a simpler equivalent for Go regex.
	// Original: \s+(?!\S) -> \s+
	content := string(raw)
	content = strings.ReplaceAll(content, `\s+(?!\S)`, `\s+`)
	content = strings.ReplaceAll(content, `\\s+(?!\\S)`, `\\s+`)

	dir := filepath.Dir(origPath)
	sanitizedPath := filepath.Join(dir, "tokenizer_sanitized.json")
	if err := os.WriteFile(sanitizedPath, []byte(content), 0o644); err != nil {
		return "", err
	}
	return sanitizedPath, nil
}

// modelFilesList returns either the defaults or overrides from MODEL_FILES env (comma-separated).
func modelFilesList(defaults []string) []string {
	val := os.Getenv("MODEL_FILES")
	if val == "" {
		return defaults
	}
	parts := strings.Split(val, ",")
	var out []string
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p != "" {
			out = append(out, p)
		}
	}
	if len(out) == 0 {
		return defaults
	}
	return out
}
