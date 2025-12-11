package transformers

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
)

// HFHubDownload downloads a file from a Hugging Face repo into a local cache.
// Very simple v1: no auth, no revision. Cache dir can be overridden with CACHE_DIR env;
// default: ./models/huggingface.co/<repoID>/resolve/main/
func HFHubDownload(repoID, filename string) (string, error) {
	cacheDir, err := hfCacheDir(repoID)
	if err != nil {
		return "", err
	}
	localPath := filepath.Join(cacheDir, filename)
	if err := os.MkdirAll(filepath.Dir(localPath), 0o755); err != nil {
		return "", err
	}

	if _, err := os.Stat(localPath); err == nil {
		// already cached
		return localPath, nil
	}

	url := fmt.Sprintf("https://huggingface.co/%s/resolve/main/%s", repoID, filename)

	if err := headURL(url); err != nil {
		return "", fmt.Errorf("HFHubDownload HEAD %s: %w", filename, err)
	}
	if err := downloadURL(url, localPath); err != nil {
		return "", fmt.Errorf("HFHubDownload GET %s: %w", filename, err)
	}

	return localPath, nil
}

// HFHubEnsureFiles checks (via HEAD) and downloads a set of files into the cache.
// Returns a map of filename -> local path.
func HFHubEnsureFiles(repoID string, files []string) (map[string]string, error) {
	cacheDir, err := hfCacheDir(repoID)
	if err != nil {
		return nil, err
	}
	res := make(map[string]string, len(files))
	for _, name := range files {
		if name == "" {
			continue
		}
		localPath := filepath.Join(cacheDir, name)
		if err := os.MkdirAll(filepath.Dir(localPath), 0o755); err != nil {
			return nil, err
		}
		if _, err := os.Stat(localPath); err == nil {
			res[name] = localPath
			continue
		}
		url := fmt.Sprintf("https://huggingface.co/%s/resolve/main/%s", repoID, name)
		if err := headURL(url); err != nil {
			return nil, fmt.Errorf("HEAD %s: %w", name, err)
		}
		if err := downloadURL(url, localPath); err != nil {
			return nil, fmt.Errorf("GET %s: %w", name, err)
		}
		res[name] = localPath
	}
	return res, nil
}

// HFHubEnsureOptionalFiles is like HFHubEnsureFiles but skips files that return 404 on HEAD.
// It returns a map of filename -> local path for the files that were found/downloaded.
func HFHubEnsureOptionalFiles(repoID string, files []string) (map[string]string, error) {
	cacheDir, err := hfCacheDir(repoID)
	if err != nil {
		return nil, err
	}
	res := make(map[string]string)
	for _, name := range files {
		if name == "" {
			continue
		}
		localPath := filepath.Join(cacheDir, name)
		if err := os.MkdirAll(filepath.Dir(localPath), 0o755); err != nil {
			return nil, err
		}
		if _, err := os.Stat(localPath); err == nil {
			res[name] = localPath
			continue
		}
		url := fmt.Sprintf("https://huggingface.co/%s/resolve/main/%s", repoID, name)
		status, err := headURLStatus(url)
		if err != nil {
			return nil, fmt.Errorf("HEAD %s: %w", name, err)
		}
		if status == http.StatusNotFound {
			continue
		}
		if status != http.StatusOK {
			return nil, fmt.Errorf("HEAD %s: status %d", name, status)
		}
		if err := downloadURL(url, localPath); err != nil {
			return nil, fmt.Errorf("GET %s: %w", name, err)
		}
		res[name] = localPath
	}
	return res, nil
}

func hfCacheDir(repoID string) (string, error) {
	base := os.Getenv("CACHE_DIR")
	if base == "" {
		base = filepath.Join(".", "models")
	}
	cacheDir := filepath.Join(base, "huggingface.co", repoID, "resolve", "main")
	if err := os.MkdirAll(cacheDir, 0o755); err != nil {
		return "", err
	}
	return cacheDir, nil
}

func headURL(url string) error {
	req, err := http.NewRequest(http.MethodHead, url, nil)
	if err != nil {
		return err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("status %d", resp.StatusCode)
	}
	return nil
}

func headURLStatus(url string) (int, error) {
	req, err := http.NewRequest(http.MethodHead, url, nil)
	if err != nil {
		return 0, err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()
	return resp.StatusCode, nil
}

func downloadURL(url, dest string) error {
	if err := os.MkdirAll(filepath.Dir(dest), 0o755); err != nil {
		return err
	}
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("status %d", resp.StatusCode)
	}
	f, err := os.Create(dest)
	if err != nil {
		return err
	}
	defer f.Close()
	if _, err := io.Copy(f, resp.Body); err != nil {
		return err
	}
	return nil
}
