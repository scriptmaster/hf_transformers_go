package transformers

import (
	"archive/tar"
	"archive/zip"
	"compress/gzip"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	onnx "github.com/yalue/onnxruntime_go"
)

// EnsureONNXRuntimeSharedLib downloads (if needed) and sets the path to the
// platform-appropriate ONNX Runtime shared library. It returns the absolute
// path configured via onnx.SetSharedLibraryPath.
func EnsureONNXRuntimeSharedLib() (string, error) {
	if path := os.Getenv("ONNXRUNTIME_SHARED_LIBRARY_PATH"); path != "" {
		if fileExists(path) {
			onnx.SetSharedLibraryPath(path)
			return path, nil
		}
	}

	spec, err := platformSpecFor(runtime.GOOS, runtime.GOARCH)
	if err != nil {
		return "", err
	}

	// Look for an already-downloaded copy in project-local cache.
	cacheDir := filepath.Join(".onnxruntime", spec.cacheDirName())
	if err := os.MkdirAll(cacheDir, 0o755); err != nil {
		return "", fmt.Errorf("create cache dir: %w", err)
	}

	if path, ok := findExistingLib(cacheDir, spec.libNames); ok {
		onnx.SetSharedLibraryPath(path)
		return path, nil
	}
	if path, ok := findExistingLib(".", spec.libNames); ok {
		onnx.SetSharedLibraryPath(path)
		return path, nil
	}

	// Download and extract.
	archivePath := filepath.Join(cacheDir, spec.archiveFilename())
	if !fileExists(archivePath) {
		if err := downloadFile(spec.url, archivePath); err != nil {
			return "", fmt.Errorf("download onnxruntime (%s): %w", spec.url, err)
		}
	}

	extractDir := filepath.Join(cacheDir, "extracted")
	_ = os.RemoveAll(extractDir)
	if err := os.MkdirAll(extractDir, 0o755); err != nil {
		return "", fmt.Errorf("create extract dir: %w", err)
	}

	if err := extractArchive(archivePath, extractDir); err != nil {
		return "", fmt.Errorf("extract archive: %w", err)
	}

	path, ok := findExistingLib(extractDir, spec.libNames)
	if !ok {
		return "", fmt.Errorf("onnxruntime library not found after extract; looked for %v", spec.libNames)
	}

	onnx.SetSharedLibraryPath(path)
	return path, nil
}

// --- helpers ---

const ortVersion = "1.22.0"
const releaseBase = "https://github.com/microsoft/onnxruntime/releases/download"

type platformSpec struct {
	url      string
	libNames []string
	arch     string
	os       string
}

func (p platformSpec) cacheDirName() string {
	return fmt.Sprintf("%s-%s-%s", p.os, p.arch, ortVersion)
}

func (p platformSpec) archiveFilename() string {
	parts := strings.Split(p.url, "/")
	return parts[len(parts)-1]
}

func platformSpecFor(goos, goarch string) (platformSpec, error) {
	type key struct{ os, arch string }
	baseURL := fmt.Sprintf("%s/v%s", releaseBase, ortVersion)
	specs := map[key]platformSpec{
		{os: "linux", arch: "amd64"}: {
			url:      fmt.Sprintf("%s/onnxruntime-linux-x64-%s.tgz", baseURL, ortVersion),
			libNames: []string{"libonnxruntime.so", "libonnxruntime.so." + ortVersion},
			arch:     "x64",
			os:       "linux",
		},
		{os: "linux", arch: "arm64"}: {
			url:      fmt.Sprintf("%s/onnxruntime-linux-aarch64-%s.tgz", baseURL, ortVersion),
			libNames: []string{"libonnxruntime.so", "libonnxruntime.so." + ortVersion},
			arch:     "aarch64",
			os:       "linux",
		},
		{os: "darwin", arch: "amd64"}: {
			url:      fmt.Sprintf("%s/onnxruntime-osx-x64-%s.tgz", baseURL, ortVersion),
			libNames: []string{"libonnxruntime.dylib"},
			arch:     "x64",
			os:       "osx",
		},
		{os: "darwin", arch: "arm64"}: {
			url:      fmt.Sprintf("%s/onnxruntime-osx-arm64-%s.tgz", baseURL, ortVersion),
			libNames: []string{"libonnxruntime.dylib"},
			arch:     "arm64",
			os:       "osx",
		},
		{os: "windows", arch: "amd64"}: {
			url:      fmt.Sprintf("%s/onnxruntime-win-x64-%s.zip", baseURL, ortVersion),
			libNames: []string{"onnxruntime.dll"},
			arch:     "x64",
			os:       "win",
		},
	}

	spec, ok := specs[key{os: goos, arch: goarch}]
	if !ok {
		return platformSpec{}, fmt.Errorf("unsupported platform: %s/%s; please set ONNXRUNTIME_SHARED_LIBRARY_PATH manually", goos, goarch)
	}
	return spec, nil
}

func findExistingLib(root string, names []string) (string, bool) {
	var found string
	filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return nil
		}
		base := filepath.Base(path)
		for _, name := range names {
			if base == name || strings.Contains(base, name) {
				found = path
				return errors.New("done")
			}
		}
		return nil
	})
	return found, found != ""
}

func downloadFile(url, dest string) error {
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	req.Header.Set("User-Agent", "github.com/scriptmaster/hf_transformers_go")
	client := &http.Client{Timeout: 5 * time.Minute}
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("unexpected status %s", resp.Status)
	}

	f, err := os.Create(dest)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = io.Copy(f, resp.Body)
	return err
}

func extractArchive(path, destDir string) error {
	if strings.HasSuffix(path, ".zip") {
		return extractZip(path, destDir)
	}
	return extractTarGz(path, destDir)
}

func extractTarGz(path, destDir string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	gzr, err := gzip.NewReader(f)
	if err != nil {
		return err
	}
	defer gzr.Close()
	tr := tar.NewReader(gzr)
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
		target := filepath.Join(destDir, hdr.Name)
		switch hdr.Typeflag {
		case tar.TypeDir:
			if err := os.MkdirAll(target, 0o755); err != nil {
				return err
			}
		case tar.TypeReg:
			if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
				return err
			}
			out, err := os.Create(target)
			if err != nil {
				return err
			}
			if _, err := io.Copy(out, tr); err != nil {
				out.Close()
				return err
			}
			out.Close()
		}
	}
	return nil
}

func extractZip(path, destDir string) error {
	r, err := zip.OpenReader(path)
	if err != nil {
		return err
	}
	defer r.Close()
	for _, f := range r.File {
		target := filepath.Join(destDir, f.Name)
		if f.FileInfo().IsDir() {
			if err := os.MkdirAll(target, 0o755); err != nil {
				return err
			}
			continue
		}
		if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
			return err
		}
		rc, err := f.Open()
		if err != nil {
			return err
		}
		out, err := os.Create(target)
		if err != nil {
			rc.Close()
			return err
		}
		if _, err := io.Copy(out, rc); err != nil {
			rc.Close()
			out.Close()
			return err
		}
		rc.Close()
		out.Close()
	}
	return nil
}

func fileExists(path string) bool {
	info, err := os.Stat(path)
	return err == nil && !info.IsDir()
}
