package transformers

import (
	"fmt"
	"os"
	"strings"
	"testing"
	"time"
)

const testModelID = "onnx-community/LFM2-350M-ONNX"

var (
	testGen        Generator
	loadDuration   time.Duration
	warmupDuration time.Duration
	secondDuration time.Duration
)

func TestMain(m *testing.M) {
	start := time.Now()
	gen, err := Pipeline(
		"text-generation",
		testModelID,
		map[string]any{"dtype": "q4"},
	)
	loadDuration = time.Since(start)
	if err != nil {
		fmt.Println("Pipeline setup error:", err)
		os.Exit(1)
	}
	testGen = gen

	warmMessages := []ChatMessage{
		{Role: RoleSystem, Content: "You are a helpful assistant."},
		{Role: RoleUser, Content: "Say warmup done."},
	}

	warmStart := time.Now()
	_, err = testGen(warmMessages, map[string]any{
		"max_new_tokens": 16,
		"do_sample":      false,
	})
	warmupDuration = time.Since(warmStart)
	if err != nil {
		fmt.Println("Warmup error:", err)
		os.Exit(1)
	}

	secondStart := time.Now()
	_, err = testGen(warmMessages, map[string]any{
		"max_new_tokens": 16,
		"do_sample":      false,
	})
	secondDuration = time.Since(secondStart)
	if err != nil {
		fmt.Println("Second-call error:", err)
		os.Exit(1)
	}

	fmt.Printf(
		"\n[LFM2 timings]\nload+init=%s\nwarmup=%s\nsecond=%s\n\n",
		loadDuration, warmupDuration, secondDuration,
	)

	os.Exit(m.Run())
}

func TestPipeline_QA(t *testing.T) {
	tests := []struct {
		name     string
		messages []ChatMessage
		expect   []string
	}{
		{
			name: "capital of France",
			messages: []ChatMessage{
				{Role: RoleSystem, Content: "You are a helpful assistant."},
				{Role: RoleUser, Content: "What is the capital of France?"},
			},
			expect: []string{"paris"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			out, err := testGen(tt.messages, map[string]any{
				"max_new_tokens": 32,
				"do_sample":      false,
			})
			if err != nil {
				t.Fatal(err)
			}
			gen := out[0]["generated_text"].([]map[string]any)
			content := strings.ToLower(gen[len(gen)-1]["content"].(string))
			if !strings.Contains(content, tt.expect[0]) {
				t.Fatalf("expected %q in %q", tt.expect[0], content)
			}
		})
	}
}
