all: demo

# Fetch dependencies
deps:
	go mod tidy
	go get github.com/sugarme/tokenizer
	go get github.com/yalue/onnxruntime_go

# Run the demo
demo: deps
	go run .

# Build binaries
build-linux: deps
	GOOS=linux GOARCH=amd64 go build -o dist/hf_transformers_go-linux-amd64 .

build-windows: deps
ifneq ($(GOOS),windows)
ifeq ($(WINDOWS),1)
	GOOS=windows GOARCH=amd64 CGO_ENABLED=1 go build -o dist/hf_transformers_go-windows-amd64.exe .
else
	@echo "Skipping Windows build; set WINDOWS=1 to attempt (requires Windows toolchain and onnxruntime DLLs)"
endif
else
	GOOS=windows GOARCH=amd64 CGO_ENABLED=1 go build -o dist/hf_transformers_go-windows-amd64.exe .
endif

# Build both (Linux + optional Windows)
build: build-linux build-windows

# Bump patch version in transformers/version.go
bump:
	@ver=$$(go run ./cmd/bump_version.go); \
	echo $$ver > .version.tmp; \
	echo "bumped version to $$ver"

# Release: bump version, commit, tag, push (CI builds artifacts on tag)
release: bump
	@ver=$$(cat .version.tmp); \
	git add transformers/version.go; \
	git commit -m "cicd: bump version to v$$ver" || true; \
	git tag "v$$ver"; \
	git push origin HEAD:main; \
	git push origin "v$$ver"; \
	echo "Release v$$ver tagged and pushed. CI will build artifacts."; \
	rm -f .version.tmp
