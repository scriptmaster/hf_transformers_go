package main

import (
	"fmt"
	"log"
	"os"

	"github.com/joho/godotenv"
	. "github.com/scriptmaster/hf_transformers_go/transformers"
)

var pipeline = Pipeline

func main() {
	_ = godotenv.Load(".env.local")

	if _, err := EnsureONNXRuntimeSharedLib(); err != nil {
		log.Fatal(err)
	}

	modelID := os.Getenv("MODEL_ID")
	if modelID == "" {
		modelID = "onnx-community/SmolLM-135M-ONNX"
	}

	generator, err := pipeline(
		"text-generation",
		modelID,
		map[string]any{"dtype": "q4"},
	)
	if err != nil {
		log.Fatal(err)
	}

	messages := []ChatMessage{
		{Role: RoleSystem, Content: "You are a helpful assistant."},
		{Role: RoleUser, Content: "What is the third planet in our solar system?"},
	}

	out, err := generator(messages, map[string]any{
		"max_new_tokens": 64,
		"do_sample":      false,
		"streamer": func(ev PipelineStreamEvent) bool {
			fmt.Print(ev.DeltaText)
			return !ev.Done
		},
	})
	if err != nil {
		log.Fatal(err)
	}

	gen := out[0]["generated_text"].([]map[string]any)
	fmt.Println("\n---")
	fmt.Println(gen[len(gen)-1]["content"].(string))
}
