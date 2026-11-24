"""Pinned public source acquisition with digest checks and explicit provenance."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlsplit

from .k1_protocol import file_sha256, freeze_manifest


@dataclass(frozen=True)
class SourceSpec:
    name: str
    purpose: str
    url: str
    revision: str
    expected_bytes: int | None = None
    sha256: str | None = None
    git_blob_sha1: str | None = None


def _verify(path: Path, spec: SourceSpec, maximum_bytes: int) -> dict:
    size = path.stat().st_size
    if size > maximum_bytes:
        raise ValueError("source exceeds configured download limit")
    if spec.expected_bytes is not None and size != spec.expected_bytes:
        raise ValueError(f"source size mismatch: {spec.name}: {size} != {spec.expected_bytes}")
    sha256 = file_sha256(path)
    if spec.sha256 is not None and sha256 != spec.sha256:
        raise ValueError(f"SHA256 mismatch: {spec.name}")
    blob = hashlib.sha1(f"blob {size}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            blob.update(chunk)
    if spec.git_blob_sha1 is not None and blob.hexdigest() != spec.git_blob_sha1:
        raise ValueError(f"Git blob SHA1 mismatch: {spec.name}")
    return dict(spec=asdict(spec), bytes=size, sha256=sha256, git_blob_sha1=blob.hexdigest())


def verify_source_manifest(directory: Path, manifest: Path, cfg: dict) -> dict[str, str]:
    """Read-only verification of every configured source against frozen acquisition provenance."""
    records = json.loads(manifest.read_text(encoding="utf-8"))["sources"]
    by_name = {record["spec"]["name"]: record for record in records}
    specs = [SourceSpec(**entry) for entry in cfg["files"]]
    if len(by_name) != len(records) or set(by_name) != {spec.name for spec in specs}:
        raise ValueError("source provenance inventory differs from configuration")
    result = {str(manifest.resolve()): file_sha256(manifest)}
    for spec in specs:
        path = (directory / spec.name).resolve()
        if path.parent != directory.resolve():
            raise ValueError("source path escapes download directory")
        verified = _verify(path, spec, cfg["maximum_download_bytes"])
        if verified != by_name[spec.name]:
            raise ValueError(f"source provenance changed: {spec.name}")
        result[str(path)] = verified["sha256"]
    return result


def download_source(spec: SourceSpec, directory: Path, session, maximum_bytes: int) -> dict:
    if Path(spec.name).name != spec.name or spec.name in ("", ".", ".."):
        raise ValueError("source name must be a plain file name")
    url = urlsplit(spec.url)
    if url.scheme != "https" or not url.hostname or url.username or url.password or not spec.revision:
        raise ValueError("a pinned public HTTPS source is required")
    if maximum_bytes <= 0:
        raise ValueError("positive download limit required")
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / spec.name
    if target.resolve().parent != directory:
        raise ValueError("source path escapes download directory")
    sidecar = directory / (spec.name + ".provenance.json")
    if target.exists():
        result = _verify(target, spec, maximum_bytes)
        if sidecar.exists():
            freeze_manifest(sidecar, result)
        elif spec.sha256 is None and spec.git_blob_sha1 is None:
            raise ValueError("existing source has no trusted digest or provenance")
        return result
    temporary = directory / (spec.name + ".partial")
    count = 0
    with session.get(spec.url, stream=True, timeout=(10, 120)) as response:
        response.raise_for_status()
        with temporary.open("wb") as stream:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                count += len(chunk)
                if count > maximum_bytes:
                    raise ValueError("source exceeds configured download limit")
                stream.write(chunk)
    result = _verify(temporary, spec, maximum_bytes)
    freeze_manifest(sidecar, result)
    temporary.replace(target)
    return result
