"""
Repository acquisition and content extraction service for RepoCompass.
Combines GitHub REST API telemetry (metadata, commit log, pull requests)
with local shallow-clone filesystem inspection for source, config, and rich document files.
"""
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import pytesseract
import requests
from docx import Document
from dotenv import load_dotenv
from PIL import Image
from pypdf import PdfReader

load_dotenv()

# Standard project manifest and configuration filenames
MANIFEST_FILENAMES = frozenset({
    "requirements.txt", "pyproject.toml", "setup.py", "setup.cfg", "Pipfile",
    "package.json", "tsconfig.json", "go.mod", "Cargo.toml", "Gemfile",
    "pom.xml", "build.gradle", "Makefile", "CMakeLists.txt", "Dockerfile",
    "docker-compose.yml", "docker-compose.yaml", ".env.example", "tox.ini", "pytest.ini",
})

# Supported programming language source file extensions
CODE_FILE_EXTENSIONS = frozenset({
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs",
    ".cpp", ".c", ".h", ".cs", ".rb", ".php", ".swift", ".kt",
})

# Rich document and image formats supported via text/OCR extraction
PARSEABLE_MEDIA_EXTENSIONS = frozenset({
    ".pdf", ".docx", ".doc",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp",
})

# Directories excluded from repository traversal
IGNORED_DIRECTORIES = frozenset({
    ".git", "node_modules", "__pycache__", ".venv", "venv", "env",
    "dist", "build", ".egg-info", ".tox", ".mypy_cache", ".pytest_cache",
    ".next", ".nuxt", "vendor", "target", ".idea", ".vscode",
})

# Non-text binary artifacts skipped during content indexing
UNSUPPORTED_BINARY_EXTENSIONS = frozenset({
    ".ico", ".svg", ".zip", ".tar", ".gz", ".rar", ".7z",
    ".exe", ".dll", ".so", ".dylib", ".class", ".jar",
    ".pyc", ".pyo", ".whl", ".egg", ".mp3", ".mp4", ".wav",
    ".avi", ".mov", ".ttf", ".woff", ".woff2", ".eot", ".sqlite", ".db",
})

README_CANDIDATES = ("README.md", "README.rst", "README.txt", "README")

# Backward-compatible module constants
CONFIG_FILE_NAMES = set(MANIFEST_FILENAMES)
SOURCE_EXTENSIONS = set(CODE_FILE_EXTENSIONS)
DOCUMENT_EXTENSIONS = set(PARSEABLE_MEDIA_EXTENSIONS)
SKIP_DIRS = set(IGNORED_DIRECTORIES)
BINARY_EXTENSIONS = set(UNSUPPORTED_BINARY_EXTENSIONS)


class GitHubClient:
    """Service client for fetching GitHub API metadata and indexing cloned repository files."""

    API_ROOT = "https://api.github.com"

    def __init__(self, token: Optional[str] = None):
        self.token = token or os.getenv("GITHUB_TOKEN")
        self.base_url = self.API_ROOT
        self.headers = {"Accept": "application/vnd.github.v3+json"}
        if self.token:
            self.headers["Authorization"] = f"token {self.token}"

    def _api_get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Perform an authenticated GET request against the GitHub REST API."""
        target_url = f"{self.base_url}/{endpoint.lstrip('/')}"
        resp = requests.get(target_url, headers=self.headers, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()

    # ---- Repository URL Resolution ----

    def parse_repo_url(self, url: str) -> Tuple[str, str]:
        """Parse a GitHub repository URL into an (owner, repository_name) tuple."""
        cleaned = url.strip()
        if "github.com" in cleaned:
            tail = cleaned.split("github.com/")[-1].strip("/")
        else:
            parsed = urlparse(cleaned)
            tail = parsed.path.strip("/")

        segments = [seg for seg in tail.split("/") if seg]
        if len(segments) < 2:
            raise ValueError(f"Unable to extract owner and repository from URL: {url}")

        owner, repo = segments[0], segments[1]
        if repo.endswith(".git"):
            repo = repo[:-4]
        return owner, repo

    # ---- GitHub REST API Queries ----

    def get_repo_metadata(self, owner: str, repo: str) -> Dict[str, Any]:
        """Retrieve high-level repository details from the GitHub API."""
        payload = self._api_get(f"repos/{owner}/{repo}")
        return {
            "name": payload.get("name"),
            "full_name": payload.get("full_name"),
            "description": payload.get("description") or "No description",
            "language": payload.get("language") or "Unknown",
            "stars": payload.get("stargazers_count", 0),
            "forks": payload.get("forks_count", 0),
            "open_issues": payload.get("open_issues_count", 0),
            "created_at": payload.get("created_at"),
            "updated_at": payload.get("updated_at"),
            "homepage": payload.get("homepage"),
            "topics": payload.get("topics", []),
            "default_branch": payload.get("default_branch", "main"),
        }

    def get_recent_commits(self, owner: str, repo: str, max_commits: int = 15) -> List[Dict[str, str]]:
        """Retrieve the most recent commit records for the repository."""
        try:
            items = self._api_get(f"repos/{owner}/{repo}/commits", params={"per_page": max_commits})
            history: List[Dict[str, str]] = []
            for entry in items:
                commit_Obj = entry.get("commit", {})
                author_Obj = commit_Obj.get("author", {})
                headline = (commit_Obj.get("message") or "").splitlines()[0][:120]
                history.append({
                    "sha": (entry.get("sha") or "")[:7],
                    "message": headline,
                    "author": author_Obj.get("name", "unknown"),
                    "date": (author_Obj.get("date") or "")[:10],
                })
            return history
        except Exception:
            return []

    def get_pull_requests(self, owner: str, repo: str, max_prs: int = 10) -> List[Dict[str, Any]]:
        """Retrieve recently updated pull requests (open and closed)."""
        query_params = {
            "state": "all",
            "per_page": max_prs,
            "sort": "updated",
            "direction": "desc",
        }
        try:
            items = self._api_get(f"repos/{owner}/{repo}/pulls", params=query_params)
            records: List[Dict[str, Any]] = []
            for item in items:
                records.append({
                    "number": item.get("number"),
                    "title": (item.get("title") or "")[:120],
                    "state": item.get("state", "unknown"),
                    "author": (item.get("user") or {}).get("login", "unknown"),
                    "created": (item.get("created_at") or "")[:10],
                    "labels": [lbl.get("name", "") for lbl in item.get("labels", [])],
                })
            return records
        except Exception:
            return []

    # ---- Shallow Clone & Local Filesystem Operations ----

    def clone_repo(self, repo_url: str) -> str:
        """Create a temporary shallow clone (`--depth 1`) of the target repository."""
        workspace_dir = tempfile.mkdtemp(prefix="repocompass_")
        proc = subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, workspace_dir],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if proc.returncode != 0:
            shutil.rmtree(workspace_dir, ignore_errors=True)
            raise RuntimeError(f"Repository clone failed: {proc.stderr.strip()}")
        return workspace_dir

    def cleanup_clone(self, repo_dir: str) -> None:
        """Delete a temporary cloned workspace directory."""
        if repo_dir and os.path.exists(repo_dir):
            shutil.rmtree(repo_dir, ignore_errors=True)

    def walk_local_repo(self, repo_dir: str) -> List[Dict[str, Any]]:
        """Traverse the cloned repository and return a normalized list of file descriptors."""
        discovered_files: List[Dict[str, Any]] = []
        base_path = Path(repo_dir)

        for current_root, subdirs, filenames in os.walk(repo_dir):
            subdirs[:] = sorted(
                d for d in subdirs if d not in IGNORED_DIRECTORIES and not d.startswith(".")
            )
            for fname in sorted(filenames):
                if fname.startswith("."):
                    continue
                abs_file = Path(current_root) / fname
                rel_posix = abs_file.relative_to(base_path).as_posix()
                try:
                    byte_size = abs_file.stat().st_size
                except OSError:
                    byte_size = 0

                discovered_files.append({
                    "path": rel_posix,
                    "type": "blob",
                    "size": byte_size,
                })

        return discovered_files

    # ---- File & Document Content Readers ----

    def read_local_file(self, repo_dir: str, file_path: str, max_lines: int = 500) -> Optional[str]:
        """Read up to `max_lines` of UTF-8 text from a file inside the cloned workspace."""
        target = Path(repo_dir) / file_path
        try:
            with target.open("r", encoding="utf-8", errors="ignore") as handle:
                collected: List[str] = []
                for idx, line in enumerate(handle):
                    if idx >= max_lines:
                        collected.append(f"\n... [truncated at {max_lines} lines]")
                        break
                    collected.append(line)
                return "".join(collected)
        except (OSError, UnicodeDecodeError):
            return None

    def extract_pdf_text(self, repo_dir: str, file_path: str, max_pages: int = 20) -> Optional[str]:
        """Extract textual content from a PDF document up to `max_pages`."""
        target = Path(repo_dir) / file_path
        try:
            pdf = PdfReader(str(target))
            total_pages = len(pdf.pages)
            limit = min(total_pages, max_pages)
            chunks: List[str] = []

            for page_idx in range(limit):
                extracted = pdf.pages[page_idx].extract_text()
                if extracted:
                    chunks.append(f"--- Page {page_idx + 1} ---\n{extracted}\n")

            if total_pages > max_pages:
                chunks.append(f"\n... [truncated: {total_pages - max_pages} additional pages]")

            return "".join(chunks) if chunks else None
        except Exception:
            return None

    def extract_docx_text(self, repo_dir: str, file_path: str) -> Optional[str]:
        """Extract non-empty paragraphs from a Word `.docx` document."""
        target = Path(repo_dir) / file_path
        try:
            word_doc = Document(str(target))
            blocks = [p.text for p in word_doc.paragraphs if p.text.strip()]
            return "\n".join(blocks) if blocks else None
        except Exception:
            return None

    def extract_image_text(self, repo_dir: str, file_path: str) -> Optional[str]:
        """Perform Tesseract OCR on an image file and return recognized text."""
        target = Path(repo_dir) / file_path
        try:
            with Image.open(target) as img:
                ocr_output = pytesseract.image_to_string(img).strip()
                return ocr_output if ocr_output else None
        except Exception:
            return None

    def read_document_file(self, repo_dir: str, file_path: str) -> Optional[str]:
        """Route a document or image file to the appropriate text extractor."""
        suffix = Path(file_path).suffix.lower()
        if suffix == ".pdf":
            return self.extract_pdf_text(repo_dir, file_path)
        if suffix in {".docx", ".doc"}:
            return self.extract_docx_text(repo_dir, file_path)
        if suffix in {".png", ".jpg", ".jpeg", ".gif", ".bmp"}:
            return self.extract_image_text(repo_dir, file_path)
        return None

    def read_all_source_files(
        self, repo_dir: str, file_tree: List[Dict[str, Any]], max_lines: int = 500
    ) -> Dict[str, str]:
        """Collect source code and parseable document contents across the repository."""
        indexed_contents: Dict[str, str] = {}

        for descriptor in file_tree:
            rel_path = descriptor["path"]
            suffix = Path(rel_path).suffix.lower()

            if suffix in UNSUPPORTED_BINARY_EXTENSIONS:
                continue
            if descriptor.get("size", 0) > 200_000:
                continue

            if suffix in CODE_FILE_EXTENSIONS:
                text = self.read_local_file(repo_dir, rel_path, max_lines=max_lines)
                if text:
                    indexed_contents[rel_path] = text
            elif suffix in PARSEABLE_MEDIA_EXTENSIONS:
                doc_text = self.read_document_file(repo_dir, rel_path)
                if doc_text:
                    indexed_contents[rel_path] = doc_text

        return indexed_contents

    def read_local_config_files(
        self, repo_dir: str, file_tree: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """Read dependency manifests and project configuration files."""
        manifests: Dict[str, str] = {}
        for descriptor in file_tree:
            rel_path = descriptor["path"]
            basename = Path(rel_path).name
            if basename in MANIFEST_FILENAMES:
                content = self.read_local_file(repo_dir, rel_path, max_lines=150)
                if content:
                    manifests[rel_path] = content
        return manifests

    def read_local_readme(self, repo_dir: str) -> Optional[str]:
        """Locate and read the primary README file from the repository root."""
        for candidate in README_CANDIDATES:
            content = self.read_local_file(repo_dir, candidate, max_lines=500)
            if content:
                return content
        return None
