package transformers

import (
	"os"

	pongo "github.com/flosch/pongo2/v6"
)

// loadChatTemplate attempts to load chat_template.jinja from the HF repo.
// Returns nil if not present or on error.
func loadChatTemplate(modelID string) (func([]ChatMessage) (string, error), error) {
	raw := []byte(defaultChatTemplateJinja)

	// Best effort fetch; do not error on absence.
	if paths, err := HFHubEnsureOptionalFiles(modelID, []string{"chat_template.jinja"}); err == nil {
		if path, ok := paths["chat_template.jinja"]; ok {
			if b, err := os.ReadFile(path); err == nil && len(b) > 0 {
				raw = b
			}
		}
	}

	tpl, err := pongo.FromString(string(raw))
	if err != nil {
		return nil, nil
	}

	renderer := func(msgs []ChatMessage) (string, error) {
		jmsgs := make([]map[string]any, 0, len(msgs))
		for _, m := range msgs {
			jmsgs = append(jmsgs, map[string]any{
				"role":    string(m.Role),
				"content": m.Content,
			})
		}
		out, err := tpl.Execute(pongo.Context{
			"messages":               jmsgs,
			"add_generation_prompt":  true,
		})
		return out, err
	}
	return renderer, nil
}

// Default Jinja chat template inspired by LFM2-350M-Extract-ONNX.
const defaultChatTemplateJinja = `{% for message in messages %}
{% if message.role == "system" %}{{ message.content }}
{% elif message.role == "user" %}User: {{ message.content }}
{% elif message.role == "assistant" %}Assistant: {{ message.content }}
{% endif %}{% endfor %}{% if add_generation_prompt %}Assistant:{% endif %}`

