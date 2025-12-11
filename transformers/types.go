package transformers

type MessageRole string

const (
	RoleSystem    MessageRole = "system"
	RoleUser      MessageRole = "user"
	RoleAssistant MessageRole = "assistant"
	RoleTool      MessageRole = "tool"
)

type ChatMessage struct {
	Role       MessageRole `json:"role"`
	Content    string      `json:"content"`
	Name       string      `json:"name,omitempty"`
	ToolCallID string      `json:"tool_call_id,omitempty"`
}

// Tool schema types – kept for future use; v1 doesn't yet embed tools into prompt.
type ToolParameter struct {
	Type        string                   `json:"type"`
	Description string                   `json:"description,omitempty"`
	Enum        []string                 `json:"enum,omitempty"`
	Properties  map[string]ToolParameter `json:"properties,omitempty"`
	Required    []string                 `json:"required,omitempty"`
}

type ToolDefinition struct {
	Name        string        `json:"name"`
	Description string        `json:"description,omitempty"`
	Parameters  ToolParameter `json:"parameters"`
}

// Streamer event exposed to user callbacks when using "streamer" option.
type PipelineStreamEvent struct {
	TokenID   int64
	DeltaText string
	FullText  string
	Step      int
	Done      bool
}

// Generator is what Pipeline(...) returns.
// It mirrors the JS/Python pattern: generator(messages, options) -> output.
type Generator func(
	messages []ChatMessage,
	options map[string]any,
) ([]map[string]any, error)
