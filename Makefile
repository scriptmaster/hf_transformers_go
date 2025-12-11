all: demo

# Fetch dependencies
deps:
	go mod tidy
	go get github.com/sugarme/tokenizer
	go get github.com/yalue/onnxruntime_go

# Run the demo
demo: deps
	go run .
