package main

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

const versionFile = "transformers/version.go"

func main() {
	cur, err := readVersion(versionFile)
	if err != nil {
		fail(err)
	}
	next, err := bumpPatch(cur)
	if err != nil {
		fail(err)
	}
	if err := writeVersion(versionFile, next); err != nil {
		fail(err)
	}
	fmt.Println(next)
}

func readVersion(path string) (string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	re := regexp.MustCompile(`const Version = "([^"]+)"`)
	m := re.FindStringSubmatch(string(data))
	if len(m) < 2 {
		return "", fmt.Errorf("version not found in %s", path)
	}
	return m[1], nil
}

func bumpPatch(ver string) (string, error) {
	parts := strings.Split(ver, ".")
	if len(parts) != 3 {
		return "", fmt.Errorf("unexpected version format: %s", ver)
	}
	parts[2] = inc(parts[2])
	return strings.Join(parts, "."), nil
}

func inc(s string) string {
	n := 0
	for _, ch := range s {
		if ch < '0' || ch > '9' {
			return s
		}
		n = n*10 + int(ch-'0')
	}
	n++
	return fmt.Sprintf("%d", n)
}

func writeVersion(path, ver string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	re := regexp.MustCompile(`const Version = "[^"]+"`)
	newData := re.ReplaceAllString(string(data), fmt.Sprintf(`const Version = "%s"`, ver))
	tmp := filepath.Join(filepath.Dir(path), ".version.tmp.write")
	if err := os.WriteFile(tmp, []byte(newData), 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

func fail(err error) {
	fmt.Fprintln(os.Stderr, "bump_version:", err)
	os.Exit(1)
}

