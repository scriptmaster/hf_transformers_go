# hf_transformers_go

Pure Go implementation of Hugging Face-style transformers.

## Quick compare: Python vs Go

Python (transformers.js-like usage):
```python
from transformers import pipeline
generator = pipeline("text-generation", model="onnx-community/SmolLM2-135M-ONNX")
out = generator("What is the third planet in our solar system and who inhabits it?", max_new_tokens=32)
print(out[0]["generated_text"])
```

Go (this repo):
```go
package main

import (
	"fmt"
	"log"

	. "github.com/scriptmaster/hf_transformers_go/transformers"
)

var pipeline = Pipeline

func main() {
	generator, err := pipeline(
		"text-generation",
		"onnx-community/SmolLM2-135M-ONNX",
		map[string]any{"dtype": "q4"},
	)
	if err != nil {
		log.Println(err)
		return
	}
	out, err := generator([]ChatMessage{
		{Role: RoleUser, Content: "What is the third planet in our solar system and who inhabits it?"},
	}, map[string]any{"max_new_tokens": 32})
	if err != nil {
		log.Println(err)
		return
	}
	fmt.Println(out[0]["generated_text"])
}
```

Notes:
- We auto-download `config.json`, `tokenizer.json`, ONNX weights (and `.onnx_data`), and optional tokenizer assets into `./models/huggingface.co/<MODEL_ID>/resolve/main/` (or `CACHE_DIR` if set).
- `generation_config.json` is parsed (if present) for eos/bos/pad IDs and default stop strings; you can also pass `stop` in call options.
- `MODEL_FILES` env can override optional asset list (comma-separated).

