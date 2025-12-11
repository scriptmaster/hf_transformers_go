# HF assets and configs

- **What we download**: by default we fetch `config.json`, `tokenizer.json`, and optional tokenizer assets (`tokenizer_config.json`, `special_tokens_map.json`, `vocab.json`, `merges.txt`). The env var `MODEL_FILES` (comma-separated) can override that optional list.
- **Where they go**: `./models/huggingface.co/<MODEL_ID>/resolve/main/<original-path>` unless `CACHE_DIR` is set.
- **ONNX files**: `onnx/model*.onnx` and `*.onnx_data` stay under `onnx/` in the same structure.
- **generation_config.json**: parsed for `eos_token_id`, `bos_token_id`, `pad_token_id`, and can supply default `stop` strings to generation. If present, it augments `config.json` values.

