#!/usr/bin/env python3
"""tensorguard — AI Model File Inspector & Security Scanner

Inspect safetensors and GGUF model files without installing ML libraries.
Parse headers, show tensor metadata, detect security risks, compare files.

Zero dependencies. Python 3.9+ stdlib only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


__version__ = "0.1.0"


# ─── Data Types ─────────────────────────────────────────────────────


SAFETENSORS_DTYPES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E5M2": 1,
    "F8_E4M3": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "F64": 8,
    "I64": 8,
    "U64": 8,
}

GGUF_MAGIC = b"GGUF"

GGUF_TYPES = {
    0: ("UINT8", 1, "B"),
    1: ("INT8", 1, "b"),
    2: ("UINT16", 2, "<H"),
    3: ("INT16", 2, "<h"),
    4: ("UINT32", 4, "<I"),
    5: ("INT32", 4, "<i"),
    6: ("FLOAT32", 4, "<f"),
    7: ("BOOL", 1, "?"),
    8: ("STRING", 0, None),  # variable length
    9: ("ARRAY", 0, None),   # variable length
    10: ("UINT64", 8, "<Q"),
    11: ("INT64", 8, "<q"),
    12: ("FLOAT64", 8, "<d"),
}

GGML_TYPES = {
    0: ("F32", 4, 1),
    1: ("F16", 2, 1),
    2: ("Q4_0", 18, 32),
    3: ("Q4_1", 20, 32),
    6: ("Q5_0", 22, 32),
    7: ("Q5_1", 24, 32),
    8: ("Q8_0", 34, 32),
    9: ("Q8_1", 40, 32),
    10: ("Q2_K", 256, 256),
    11: ("Q3_K", 256, 256),
    12: ("Q4_K", 144, 256),
    13: ("Q5_K", 176, 256),
    14: ("Q6_K", 210, 256),
    15: ("Q8_K", 292, 256),
    16: ("IQ2_XXS", 66, 256),
    17: ("IQ2_XS", 74, 256),
    18: ("IQ3_XXS", 98, 256),
    19: ("IQ1_S", 50, 256),
    20: ("IQ4_NL", 18, 32),
    21: ("IQ3_S", 110, 256),
    22: ("IQ2_S", 82, 256),
    23: ("IQ4_XS", 36, 64),
    24: ("IQ1_M", 56, 256),
    25: ("BF16", 2, 1),
    26: ("Q4_0_4_4", 18, 32),
    27: ("Q4_0_4_8", 18, 32),
    28: ("Q4_0_8_8", 18, 32),
    29: ("TQ1_0", 0, 0),
    30: ("TQ2_0", 0, 0),
}


# ─── Findings ───────────────────────────────────────────────────────


@dataclass
class Finding:
    """A security-relevant issue detected in a model file."""
    severity: str      # critical, high, medium, low, info
    rule: str          # Rule ID
    title: str
    detail: str
    remediation: str = ""


@dataclass
class TensorInfo:
    """Metadata for a single tensor."""
    name: str
    dtype: str
    shape: list[int]
    offset_start: int = 0
    offset_end: int = 0

    @property
    def num_elements(self) -> int:
        result = 1
        for dim in self.shape:
            result *= dim
        return result

    @property
    def size_bytes(self) -> int:
        bytes_per_elem = SAFETENSORS_DTYPES.get(self.dtype, 4)
        return self.num_elements * bytes_per_elem


@dataclass
class ModelInfo:
    """Complete model file information."""
    path: str
    format: str  # "safetensors" or "gguf"
    file_size: int = 0
    header_size: int = 0
    tensors: list[TensorInfo] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)
    # GGUF specific
    gguf_version: int = 0
    gguf_tensor_count: int = 0
    gguf_kv_count: int = 0


# ─── Safetensors Parser ────────────────────────────────────────────


def parse_safetensors(filepath: str) -> ModelInfo:
    """Parse a safetensors file header."""
    info = ModelInfo(path=filepath, format="safetensors")
    info.file_size = os.path.getsize(filepath)

    with open(filepath, "rb") as f:
        # Read header size (8 bytes, little-endian u64)
        header_size_bytes = f.read(8)
        if len(header_size_bytes) < 8:
            info.findings.append(Finding(
                severity="critical", rule="TG01",
                title="File too small",
                detail=f"File is only {info.file_size} bytes — cannot read header size",
                remediation="File may be corrupted or not a valid safetensors file.",
            ))
            return info

        header_size = struct.unpack("<Q", header_size_bytes)[0]
        info.header_size = header_size

        # Sanity check header size
        if header_size > info.file_size - 8:
            info.findings.append(Finding(
                severity="critical", rule="TG01",
                title="Header size exceeds file",
                detail=f"Header claims {header_size:,} bytes but file is only {info.file_size:,} bytes",
                remediation="File may be corrupted or maliciously crafted.",
            ))
            return info

        if header_size > 100_000_000:  # 100MB header is suspicious
            info.findings.append(Finding(
                severity="high", rule="TG02",
                title="Extremely large header",
                detail=f"Header is {header_size:,} bytes ({header_size / 1024 / 1024:.1f} MB)",
                remediation="Large headers may contain embedded payloads. Inspect metadata carefully.",
            ))

        if header_size == 0:
            info.findings.append(Finding(
                severity="high", rule="TG01",
                title="Empty header",
                detail="Header size is 0 bytes — no tensor metadata",
                remediation="File may be corrupted.",
            ))
            return info

        # Read and parse header JSON
        header_bytes = f.read(header_size)
        if len(header_bytes) < header_size:
            info.findings.append(Finding(
                severity="critical", rule="TG01",
                title="Truncated header",
                detail=f"Expected {header_size:,} bytes but read {len(header_bytes):,}",
                remediation="File is truncated or corrupted.",
            ))
            return info

        try:
            header = json.loads(header_bytes)
        except json.JSONDecodeError as e:
            info.findings.append(Finding(
                severity="critical", rule="TG01",
                title="Invalid JSON header",
                detail=f"JSON parse error: {e}",
                remediation="Header is not valid JSON. File may be corrupted or not safetensors format.",
            ))
            return info

        if not isinstance(header, dict):
            info.findings.append(Finding(
                severity="critical", rule="TG01",
                title="Header is not a JSON object",
                detail=f"Expected object, got {type(header).__name__}",
                remediation="Safetensors header must be a JSON object mapping tensor names to info.",
            ))
            return info

        # Extract metadata
        if "__metadata__" in header:
            info.metadata = header.pop("__metadata__")
            if not isinstance(info.metadata, dict):
                info.findings.append(Finding(
                    severity="medium", rule="TG03",
                    title="Invalid __metadata__ type",
                    detail=f"Expected object, got {type(info.metadata).__name__}",
                ))
                info.metadata = {}

        # Parse tensors
        data_start = 8 + header_size
        for name, tensor_data in header.items():
            if not isinstance(tensor_data, dict):
                info.findings.append(Finding(
                    severity="medium", rule="TG04",
                    title=f"Invalid tensor entry: {name}",
                    detail=f"Expected object, got {type(tensor_data).__name__}",
                ))
                continue

            dtype = tensor_data.get("dtype", "UNKNOWN")
            shape = tensor_data.get("shape", [])
            offsets = tensor_data.get("data_offsets", [0, 0])

            tensor = TensorInfo(
                name=name,
                dtype=dtype,
                shape=shape,
                offset_start=offsets[0] if len(offsets) > 0 else 0,
                offset_end=offsets[1] if len(offsets) > 1 else 0,
            )
            info.tensors.append(tensor)

        # Run security checks
        _check_safetensors_security(info)

    return info


def _check_safetensors_security(info: ModelInfo) -> None:
    """Run security checks on parsed safetensors file."""

    # Check metadata for suspicious content
    _check_metadata_security(info)

    # Check tensor integrity
    data_start = 8 + info.header_size

    # Check for overlapping tensors
    sorted_tensors = sorted(info.tensors, key=lambda t: t.offset_start)
    for i in range(len(sorted_tensors) - 1):
        current = sorted_tensors[i]
        next_t = sorted_tensors[i + 1]
        if current.offset_end > next_t.offset_start:
            info.findings.append(Finding(
                severity="high", rule="TG05",
                title="Overlapping tensor data",
                detail=f"'{current.name}' [{current.offset_start}:{current.offset_end}] overlaps "
                       f"'{next_t.name}' [{next_t.offset_start}:{next_t.offset_end}]",
                remediation="Overlapping tensors are abnormal and may indicate data corruption or tampering.",
            ))

    # Check for tensors extending beyond file
    for t in info.tensors:
        abs_end = data_start + t.offset_end
        if abs_end > info.file_size:
            info.findings.append(Finding(
                severity="high", rule="TG06",
                title=f"Tensor extends beyond file: {t.name}",
                detail=f"Tensor data ends at byte {abs_end:,} but file is {info.file_size:,} bytes",
                remediation="File may be truncated or tensor offsets are incorrect.",
            ))

    # Check for unusual dtypes
    for t in info.tensors:
        if t.dtype not in SAFETENSORS_DTYPES:
            info.findings.append(Finding(
                severity="medium", rule="TG07",
                title=f"Unknown dtype: {t.dtype}",
                detail=f"Tensor '{t.name}' uses unrecognized dtype '{t.dtype}'",
                remediation="Unknown dtypes may cause loading errors or indicate a newer format version.",
            ))

    # Check for empty tensors
    empty_count = sum(1 for t in info.tensors if t.num_elements == 0)
    if empty_count > 0:
        info.findings.append(Finding(
            severity="low", rule="TG08",
            title=f"{empty_count} empty tensor(s)",
            detail="Tensors with 0 elements detected",
        ))

    # Check for suspicious tensor names
    _check_tensor_names(info)

    # Check for gap between header and tensor data (could hide payload)
    if info.tensors:
        min_offset = min(t.offset_start for t in info.tensors)
        if min_offset > 0:
            info.findings.append(Finding(
                severity="low", rule="TG09",
                title="Gap before tensor data",
                detail=f"{min_offset:,} bytes of unused space before first tensor",
                remediation="Padding is normal in small amounts but large gaps could hide data.",
            ))

    # Check data section for trailing bytes
    if info.tensors:
        max_end = max(t.offset_end for t in info.tensors)
        expected_file_size = 8 + info.header_size + max_end
        trailing = info.file_size - expected_file_size
        if trailing > 0:
            sev = "medium" if trailing > 1024 else "low"
            info.findings.append(Finding(
                severity=sev, rule="TG10",
                title=f"Trailing data after tensors",
                detail=f"{trailing:,} bytes after last tensor (could be padding or hidden data)",
                remediation="Small amounts of padding are normal. Large trailing sections are suspicious.",
            ))


def _check_metadata_security(info: ModelInfo) -> None:
    """Check metadata for security-relevant patterns."""
    if not info.metadata:
        return

    # Check for class references (Unit42 attack vector)
    class_patterns = [
        r'\b(?:__class__|__subclasses__|__import__|__builtins__)\b',
        r'\b(?:eval|exec|compile|getattr|setattr|delattr)\s*\(',
        r'\b(?:os\.system|subprocess|Popen|call)\b',
        r'\b(?:importlib|__reduce__|__getstate__|__setstate__)\b',
        r'\btrust_remote_code\b',
    ]

    for key, value in info.metadata.items():
        val_str = str(value)

        # Check key names
        if key.startswith("_") and key != "__metadata__":
            info.findings.append(Finding(
                severity="medium", rule="TG11",
                title=f"Underscore-prefixed metadata key: {key}",
                detail=f"Key '{key}' starts with underscore — may be internal/system metadata",
                remediation="Review underscore-prefixed metadata for unexpected content.",
            ))

        # Check values for class/code references
        for pattern in class_patterns:
            if re.search(pattern, val_str, re.IGNORECASE):
                info.findings.append(Finding(
                    severity="critical", rule="TG12",
                    title=f"Code pattern in metadata: {key}",
                    detail=f"Metadata key '{key}' contains pattern matching '{pattern}'",
                    remediation="Metadata containing code patterns may be used for code execution attacks. "
                                "See Unit42 research on safetensors metadata exploitation.",
                ))
                break

        # Check for base64 encoded content
        if isinstance(value, str) and len(value) > 100:
            # Simple heuristic: high ratio of base64-valid chars
            b64_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=")
            if len(value) > 200 and sum(1 for c in value if c in b64_chars) / len(value) > 0.95:
                info.findings.append(Finding(
                    severity="medium", rule="TG13",
                    title=f"Possible base64 payload in metadata: {key}",
                    detail=f"Metadata key '{key}' contains {len(value):,} chars of likely base64 data",
                    remediation="Base64-encoded content in metadata is unusual and may contain hidden payloads.",
                ))

        # Check for very large metadata values
        if isinstance(value, str) and len(value) > 10000:
            info.findings.append(Finding(
                severity="low", rule="TG14",
                title=f"Large metadata value: {key}",
                detail=f"Metadata key '{key}' is {len(value):,} characters",
                remediation="Unusually large metadata values may indicate embedded data.",
            ))

    # Check for suspicious metadata keys that look like configuration
    config_keys = ["auto_map", "architectures", "model_type", "custom_pipelines",
                   "pipeline_tag", "transformers_version"]
    for key in info.metadata:
        if key in config_keys:
            info.findings.append(Finding(
                severity="info", rule="TG15",
                title=f"Configuration metadata: {key}",
                detail=f"Metadata contains config key '{key}' = '{str(info.metadata[key])[:100]}'",
            ))


def _check_tensor_names(info: ModelInfo) -> None:
    """Check tensor names for suspicious patterns."""
    suspicious_patterns = [
        (r'(?:backdoor|trojan|payload|exploit|malicious|hidden)', "Suspicious keyword in tensor name"),
        (r'(?:\.exe|\.dll|\.so|\.sh|\.bat|\.cmd|\.ps1)', "Executable extension in tensor name"),
        (r'^[\da-f]{32,}$', "Hash-like tensor name (may hide data)"),
    ]

    for tensor in info.tensors:
        for pattern, desc in suspicious_patterns:
            if re.search(pattern, tensor.name, re.IGNORECASE):
                info.findings.append(Finding(
                    severity="medium", rule="TG16",
                    title=f"Suspicious tensor name: {tensor.name}",
                    detail=desc,
                    remediation="Unusual tensor names may indicate hidden data or backdoors.",
                ))
                break

        # Check for very long tensor names (could embed data)
        if len(tensor.name) > 500:
            info.findings.append(Finding(
                severity="medium", rule="TG17",
                title=f"Extremely long tensor name ({len(tensor.name)} chars)",
                detail=f"First 100 chars: {tensor.name[:100]}...",
                remediation="Long tensor names are abnormal and could be used to embed data.",
            ))


# ─── GGUF Parser ────────────────────────────────────────────────────


def parse_gguf(filepath: str) -> ModelInfo:
    """Parse a GGUF file header."""
    info = ModelInfo(path=filepath, format="gguf")
    info.file_size = os.path.getsize(filepath)

    with open(filepath, "rb") as f:
        # Magic number
        magic = f.read(4)
        if magic != GGUF_MAGIC:
            info.findings.append(Finding(
                severity="critical", rule="TG01",
                title="Invalid GGUF magic number",
                detail=f"Expected b'GGUF', got {magic!r}",
                remediation="File is not a valid GGUF file.",
            ))
            return info

        # Version
        version = struct.unpack("<I", f.read(4))[0]
        info.gguf_version = version
        if version not in (2, 3):
            info.findings.append(Finding(
                severity="medium", rule="TG18",
                title=f"Unusual GGUF version: {version}",
                detail=f"Expected version 2 or 3, got {version}",
                remediation="This may be an unsupported format version.",
            ))

        # Tensor count and KV count
        info.gguf_tensor_count = struct.unpack("<Q", f.read(8))[0]
        info.gguf_kv_count = struct.unpack("<Q", f.read(8))[0]

        if info.gguf_tensor_count > 100000:
            info.findings.append(Finding(
                severity="high", rule="TG19",
                title=f"Extreme tensor count: {info.gguf_tensor_count:,}",
                detail="Suspiciously high number of tensors",
                remediation="This may indicate a corrupted or maliciously crafted file.",
            ))

        if info.gguf_kv_count > 10000:
            info.findings.append(Finding(
                severity="medium", rule="TG19",
                title=f"Very high KV count: {info.gguf_kv_count:,}",
                detail="Unusually many key-value metadata entries",
            ))

        # Parse KV pairs
        info.header_size = f.tell()  # track where header parsing ends
        for _ in range(info.gguf_kv_count):
            try:
                key = _read_gguf_string(f)
                value_type = struct.unpack("<I", f.read(4))[0]
                value = _read_gguf_value(f, value_type)
                info.metadata[key] = value
            except (struct.error, EOFError, ValueError):
                info.findings.append(Finding(
                    severity="high", rule="TG01",
                    title="Truncated GGUF metadata",
                    detail=f"Failed to parse KV pair after reading {len(info.metadata)} entries",
                    remediation="File may be truncated or corrupted.",
                ))
                break

        # Parse tensor info
        for _ in range(info.gguf_tensor_count):
            try:
                name = _read_gguf_string(f)
                n_dims = struct.unpack("<I", f.read(4))[0]
                dims = []
                for _ in range(n_dims):
                    dims.append(struct.unpack("<Q", f.read(8))[0])
                ggml_type = struct.unpack("<I", f.read(4))[0]
                offset = struct.unpack("<Q", f.read(8))[0]

                dtype_name = GGML_TYPES.get(ggml_type, (f"UNKNOWN({ggml_type})", 0, 0))[0]
                tensor = TensorInfo(
                    name=name,
                    dtype=dtype_name,
                    shape=dims,
                    offset_start=offset,
                )
                info.tensors.append(tensor)
            except (struct.error, EOFError, ValueError):
                info.findings.append(Finding(
                    severity="high", rule="TG01",
                    title="Truncated GGUF tensor info",
                    detail=f"Failed to parse tensor info after reading {len(info.tensors)} tensors",
                    remediation="File may be truncated or corrupted.",
                ))
                break

        info.header_size = f.tell()

    # Run security checks
    _check_gguf_security(info)
    _check_tensor_names(info)

    return info


def _read_gguf_string(f) -> str:
    """Read a GGUF string (length-prefixed)."""
    length = struct.unpack("<Q", f.read(8))[0]
    if length > 10_000_000:  # 10MB string limit
        raise ValueError(f"String length {length} exceeds safety limit")
    data = f.read(length)
    if len(data) < length:
        raise EOFError("Unexpected end of file reading string")
    return data.decode("utf-8", errors="replace")


def _read_gguf_value(f, value_type: int) -> Any:
    """Read a GGUF typed value."""
    if value_type in GGUF_TYPES:
        type_name, size, fmt = GGUF_TYPES[value_type]
        if type_name == "STRING":
            return _read_gguf_string(f)
        elif type_name == "ARRAY":
            elem_type = struct.unpack("<I", f.read(4))[0]
            count = struct.unpack("<Q", f.read(8))[0]
            if count > 1_000_000:
                raise ValueError(f"Array count {count} exceeds safety limit")
            values = []
            for _ in range(count):
                values.append(_read_gguf_value(f, elem_type))
            return values
        elif fmt is not None:
            return struct.unpack(fmt, f.read(size))[0]
    raise ValueError(f"Unknown GGUF value type: {value_type}")


def _check_gguf_security(info: ModelInfo) -> None:
    """Run security checks on GGUF metadata."""
    for key, value in info.metadata.items():
        val_str = str(value)

        # Check for executable content
        exe_patterns = [
            r'\b(?:eval|exec|system|popen|subprocess)\b',
            r'\b(?:import\s+os|import\s+subprocess|from\s+os)\b',
            r'\b(?:curl\s|wget\s|nc\s|bash\s|sh\s+-c)\b',
            r'(?:python\s+-c|ruby\s+-e|perl\s+-e)\b',
        ]
        for pattern in exe_patterns:
            if re.search(pattern, val_str, re.IGNORECASE):
                info.findings.append(Finding(
                    severity="critical", rule="TG12",
                    title=f"Executable pattern in GGUF metadata: {key}",
                    detail=f"Key '{key}' contains executable content pattern",
                    remediation="GGUF metadata should not contain executable code.",
                ))
                break

        # Check for suspiciously large values
        if isinstance(value, str) and len(value) > 50000:
            info.findings.append(Finding(
                severity="medium", rule="TG14",
                title=f"Large GGUF metadata value: {key}",
                detail=f"Key '{key}' is {len(value):,} characters",
            ))

    # Check general.name for suspicious values
    general_name = info.metadata.get("general.name", "")
    if isinstance(general_name, str):
        if re.search(r'[<>|;&`$]', general_name):
            info.findings.append(Finding(
                severity="high", rule="TG20",
                title="Shell metacharacters in model name",
                detail=f"general.name contains shell-significant characters: {general_name!r}",
                remediation="Model names with shell metacharacters could be exploited if passed to commands unsafely.",
            ))


# ─── Format Detection ──────────────────────────────────────────────


def detect_format(filepath: str) -> str | None:
    """Detect model file format from magic bytes and extension."""
    ext = Path(filepath).suffix.lower()

    # Check by extension first
    if ext in (".safetensors",):
        return "safetensors"
    if ext in (".gguf",):
        return "gguf"

    # Check magic bytes
    try:
        with open(filepath, "rb") as f:
            magic = f.read(4)
            if magic == GGUF_MAGIC:
                return "gguf"
            # Safetensors starts with 8 bytes of header size, then JSON
            f.seek(0)
            header_size_bytes = f.read(8)
            if len(header_size_bytes) == 8:
                header_size = struct.unpack("<Q", header_size_bytes)[0]
                if 2 <= header_size <= 100_000_000:
                    peek = f.read(min(1, header_size))
                    if peek == b"{":
                        return "safetensors"
    except (OSError, struct.error):
        pass

    return None


def parse_model(filepath: str) -> ModelInfo:
    """Parse a model file, auto-detecting format."""
    fmt = detect_format(filepath)
    if fmt == "safetensors":
        return parse_safetensors(filepath)
    elif fmt == "gguf":
        return parse_gguf(filepath)
    else:
        info = ModelInfo(path=filepath, format="unknown")
        info.file_size = os.path.getsize(filepath) if os.path.exists(filepath) else 0
        info.findings.append(Finding(
            severity="high", rule="TG00",
            title="Unknown file format",
            detail=f"Cannot detect format for {filepath}",
            remediation="Supported formats: .safetensors, .gguf",
        ))
        return info


# ─── Statistics ─────────────────────────────────────────────────────


def compute_stats(info: ModelInfo) -> dict[str, Any]:
    """Compute model statistics."""
    stats: dict[str, Any] = {
        "format": info.format,
        "file_size": info.file_size,
        "file_size_human": _human_size(info.file_size),
        "header_size": info.header_size,
        "header_size_human": _human_size(info.header_size),
        "tensor_count": len(info.tensors),
        "metadata_keys": len(info.metadata),
    }

    if info.tensors:
        total_params = sum(t.num_elements for t in info.tensors)
        stats["total_parameters"] = total_params
        stats["total_parameters_human"] = _human_count(total_params)

        # Dtype distribution
        dtype_counts: dict[str, int] = {}
        dtype_params: dict[str, int] = {}
        for t in info.tensors:
            dtype_counts[t.dtype] = dtype_counts.get(t.dtype, 0) + 1
            dtype_params[t.dtype] = dtype_params.get(t.dtype, 0) + t.num_elements
        stats["dtype_distribution"] = dtype_counts
        stats["dtype_parameters"] = {k: _human_count(v) for k, v in dtype_params.items()}

        # Shape statistics
        max_dims = max(len(t.shape) for t in info.tensors)
        stats["max_dimensions"] = max_dims
        if info.format == "safetensors":
            total_data = sum(t.offset_end - t.offset_start for t in info.tensors)
            stats["total_data_size"] = total_data
            stats["total_data_size_human"] = _human_size(total_data)

    if info.format == "gguf":
        stats["gguf_version"] = info.gguf_version
        stats["declared_tensor_count"] = info.gguf_tensor_count
        model_type = info.metadata.get("general.architecture", "unknown")
        stats["architecture"] = model_type
        model_name = info.metadata.get("general.name", "unknown")
        stats["model_name"] = model_name
        quant_version = info.metadata.get("general.quantization_version", None)
        if quant_version is not None:
            stats["quantization_version"] = quant_version

    return stats


# ─── Comparison ─────────────────────────────────────────────────────


def compare_models(info_a: ModelInfo, info_b: ModelInfo) -> dict[str, Any]:
    """Compare two model files."""
    tensors_a = {t.name: t for t in info_a.tensors}
    tensors_b = {t.name: t for t in info_b.tensors}

    names_a = set(tensors_a.keys())
    names_b = set(tensors_b.keys())

    added = sorted(names_b - names_a)
    removed = sorted(names_a - names_b)
    common = sorted(names_a & names_b)

    changed = []
    for name in common:
        ta = tensors_a[name]
        tb = tensors_b[name]
        diffs = []
        if ta.dtype != tb.dtype:
            diffs.append(f"dtype: {ta.dtype} → {tb.dtype}")
        if ta.shape != tb.shape:
            diffs.append(f"shape: {ta.shape} → {tb.shape}")
        if diffs:
            changed.append({"name": name, "changes": diffs})

    return {
        "file_a": info_a.path,
        "file_b": info_b.path,
        "format_a": info_a.format,
        "format_b": info_b.format,
        "tensors_added": added,
        "tensors_removed": removed,
        "tensors_changed": changed,
        "tensors_unchanged": len(common) - len(changed),
        "summary": {
            "added": len(added),
            "removed": len(removed),
            "changed": len(changed),
            "unchanged": len(common) - len(changed),
        },
    }


# ─── Scoring ────────────────────────────────────────────────────────

SEVERITY_WEIGHTS = {"critical": 25, "high": 10, "medium": 3, "low": 1, "info": 0}


def calculate_grade(findings: list[Finding]) -> tuple[str, int]:
    """Calculate security grade from findings."""
    score = sum(SEVERITY_WEIGHTS.get(f.severity, 0) for f in findings)
    if score == 0:
        return "A", score
    elif score <= 5:
        return "B", score
    elif score <= 15:
        return "C", score
    elif score <= 30:
        return "D", score
    else:
        return "F", score


# ─── Formatting ─────────────────────────────────────────────────────

SEVERITY_COLORS = {
    "critical": "\033[91m",
    "high": "\033[93m",
    "medium": "\033[33m",
    "low": "\033[36m",
    "info": "\033[90m",
}
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GRADE_COLORS = {
    "A": "\033[92m", "B": "\033[93m", "C": "\033[33m",
    "D": "\033[91m", "F": "\033[91m",
}


def _human_size(size: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(size) < 1024.0:
            return f"{size:.1f} {unit}" if unit != "B" else f"{size} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"


def _human_count(count: int) -> str:
    """Format large counts (e.g., parameter counts)."""
    if count >= 1_000_000_000:
        return f"{count / 1_000_000_000:.2f}B"
    elif count >= 1_000_000:
        return f"{count / 1_000_000:.2f}M"
    elif count >= 1_000:
        return f"{count / 1_000:.1f}K"
    return str(count)


def format_inspect(info: ModelInfo, stats: dict[str, Any], verbose: bool = False,
                   no_color: bool = False) -> str:
    """Format model inspection output."""
    if no_color:
        b = r = d = ""
        sc = {k: "" for k in SEVERITY_COLORS}
        gc = {k: "" for k in GRADE_COLORS}
    else:
        b, r, d = BOLD, RESET, DIM
        sc = SEVERITY_COLORS
        gc = GRADE_COLORS

    lines = []
    lines.append(f"{b}tensorguard v{__version__} — AI Model File Inspector{r}")
    lines.append("")

    # File info
    lines.append(f"{b}FILE{r}")
    lines.append(f"  Path:     {info.path}")
    lines.append(f"  Format:   {info.format}")
    lines.append(f"  Size:     {stats['file_size_human']} ({stats['file_size']:,} bytes)")
    lines.append(f"  Header:   {stats['header_size_human']} ({stats['header_size']:,} bytes)")
    lines.append(f"  Tensors:  {stats['tensor_count']}")
    lines.append(f"  Metadata: {stats['metadata_keys']} keys")
    lines.append("")

    # Model stats
    if "total_parameters" in stats:
        lines.append(f"{b}MODEL{r}")
        lines.append(f"  Parameters: {stats['total_parameters_human']} ({stats['total_parameters']:,})")
        if "architecture" in stats:
            lines.append(f"  Architecture: {stats['architecture']}")
        if "model_name" in stats:
            lines.append(f"  Name: {stats['model_name']}")
        if "gguf_version" in stats:
            lines.append(f"  GGUF version: {stats['gguf_version']}")
        lines.append("")

    # Dtype distribution
    if "dtype_distribution" in stats:
        lines.append(f"{b}DTYPES{r}")
        for dtype, count in sorted(stats["dtype_distribution"].items()):
            params = stats.get("dtype_parameters", {}).get(dtype, "?")
            lines.append(f"  {dtype:>12}: {count:>5} tensors ({params} params)")
        lines.append("")

    # Metadata
    if info.metadata and verbose:
        lines.append(f"{b}METADATA{r}")
        for key, value in sorted(info.metadata.items()):
            val_str = str(value)
            if len(val_str) > 80:
                val_str = val_str[:77] + "..."
            lines.append(f"  {key}: {val_str}")
        lines.append("")

    # Tensors (verbose mode)
    if verbose and info.tensors:
        lines.append(f"{b}TENSORS{r}")
        for t in info.tensors[:100]:  # Limit display
            shape_str = "×".join(str(s) for s in t.shape) if t.shape else "scalar"
            lines.append(f"  {t.name:>50}  {t.dtype:>8}  [{shape_str}]")
        if len(info.tensors) > 100:
            lines.append(f"  ... and {len(info.tensors) - 100} more tensors")
        lines.append("")

    # Security findings
    grade, score = calculate_grade(info.findings)
    lines.append(f"{b}SECURITY{r}")
    lines.append(f"  Grade: {gc.get(grade, '')}{grade}{r} (risk score: {score})")
    lines.append(f"  Findings: {len(info.findings)}")

    by_sev = {}
    for f in info.findings:
        by_sev[f.severity] = by_sev.get(f.severity, 0) + 1
    parts = []
    for sev in ["critical", "high", "medium", "low", "info"]:
        if sev in by_sev:
            parts.append(f"{sc.get(sev, '')}{by_sev[sev]} {sev}{r}")
    if parts:
        lines.append(f"  Breakdown: {', '.join(parts)}")
    lines.append("")

    if info.findings:
        for f in sorted(info.findings, key=lambda x: list(SEVERITY_WEIGHTS.keys()).index(x.severity)):
            color = sc.get(f.severity, "")
            lines.append(f"  {color}[{f.severity.upper():>8}]{r} [{f.rule}] {f.title}")
            if f.detail != f.title:
                lines.append(f"             {f.detail}")
            if f.remediation and verbose:
                lines.append(f"             → {f.remediation}")
    else:
        lines.append(f"  {gc['A']}✅ No security issues detected.{r}")
    lines.append("")

    return "\n".join(lines)


def format_json_output(info: ModelInfo, stats: dict[str, Any]) -> str:
    """Format as JSON."""
    grade, score = calculate_grade(info.findings)
    data = {
        "version": __version__,
        "file": info.path,
        "format": info.format,
        "file_size": info.file_size,
        "header_size": info.header_size,
        "stats": stats,
        "metadata": {k: str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v
                     for k, v in info.metadata.items()},
        "tensors": [
            {
                "name": t.name,
                "dtype": t.dtype,
                "shape": t.shape,
                "elements": t.num_elements,
            }
            for t in info.tensors
        ],
        "security": {
            "grade": grade,
            "risk_score": score,
            "findings": [
                {
                    "severity": f.severity,
                    "rule": f.rule,
                    "title": f.title,
                    "detail": f.detail,
                    "remediation": f.remediation,
                }
                for f in info.findings
            ],
        },
    }
    return json.dumps(data, indent=2, default=str)


def format_compare(diff: dict[str, Any], no_color: bool = False) -> str:
    """Format comparison output."""
    b = BOLD if not no_color else ""
    r = RESET if not no_color else ""
    g = "\033[92m" if not no_color else ""
    red = "\033[91m" if not no_color else ""
    y = "\033[93m" if not no_color else ""

    lines = []
    lines.append(f"{b}tensorguard v{__version__} — Model Comparison{r}")
    lines.append("")
    lines.append(f"  A: {diff['file_a']} ({diff['format_a']})")
    lines.append(f"  B: {diff['file_b']} ({diff['format_b']})")
    lines.append("")

    s = diff["summary"]
    lines.append(f"{b}SUMMARY{r}")
    lines.append(f"  Added:     {g}{s['added']}{r}")
    lines.append(f"  Removed:   {red}{s['removed']}{r}")
    lines.append(f"  Changed:   {y}{s['changed']}{r}")
    lines.append(f"  Unchanged: {s['unchanged']}")
    lines.append("")

    if diff["tensors_added"]:
        lines.append(f"{b}ADDED{r}")
        for name in diff["tensors_added"][:50]:
            lines.append(f"  {g}+ {name}{r}")
        if len(diff["tensors_added"]) > 50:
            lines.append(f"  ... and {len(diff['tensors_added']) - 50} more")
        lines.append("")

    if diff["tensors_removed"]:
        lines.append(f"{b}REMOVED{r}")
        for name in diff["tensors_removed"][:50]:
            lines.append(f"  {red}- {name}{r}")
        if len(diff["tensors_removed"]) > 50:
            lines.append(f"  ... and {len(diff['tensors_removed']) - 50} more")
        lines.append("")

    if diff["tensors_changed"]:
        lines.append(f"{b}CHANGED{r}")
        for change in diff["tensors_changed"][:50]:
            lines.append(f"  {y}~ {change['name']}{r}")
            for c in change["changes"]:
                lines.append(f"    {c}")
        if len(diff["tensors_changed"]) > 50:
            lines.append(f"  ... and {len(diff['tensors_changed']) - 50} more")
        lines.append("")

    return "\n".join(lines)


# ─── Main ───────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="tensorguard",
        description="AI Model File Inspector & Security Scanner",
        epilog="Examples:\n"
               "  tensorguard model.safetensors           # Inspect safetensors file\n"
               "  tensorguard model.gguf                   # Inspect GGUF file\n"
               "  tensorguard model.safetensors --verbose  # Show all tensors & metadata\n"
               "  tensorguard --compare a.safetensors b.safetensors  # Compare models\n"
               "  tensorguard model.gguf --json            # JSON output\n"
               "  tensorguard model.safetensors --ci       # CI mode (exit 1 if unsafe)\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "files", nargs="+",
        help="Model file(s) to inspect",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Compare two model files (requires exactly 2 files)",
    )
    parser.add_argument(
        "--json", action="store_true", dest="json_output",
        help="Output as JSON",
    )
    parser.add_argument(
        "--verbose", "-V", action="store_true",
        help="Show all tensors, metadata, and remediations",
    )
    parser.add_argument(
        "--no-color", action="store_true",
        help="Disable colored output",
    )
    parser.add_argument(
        "--ci", action="store_true",
        help="CI mode: exit 1 if grade below threshold",
    )
    parser.add_argument(
        "--threshold", default="C",
        help="Minimum passing grade for CI mode (default: C)",
    )
    parser.add_argument(
        "--min-severity", default="info",
        choices=["critical", "high", "medium", "low", "info"],
        help="Minimum severity to report (default: info)",
    )
    parser.add_argument(
        "--metadata-only", action="store_true",
        help="Only show metadata (skip tensor listing)",
    )
    parser.add_argument(
        "--version", "-v", action="version",
        version=f"tensorguard {__version__}",
    )
    args = parser.parse_args()

    no_color = args.no_color or not sys.stdout.isatty()

    # Compare mode
    if args.compare:
        if len(args.files) != 2:
            print("Error: --compare requires exactly 2 files", file=sys.stderr)
            return 1
        info_a = parse_model(args.files[0])
        info_b = parse_model(args.files[1])
        diff = compare_models(info_a, info_b)

        if args.json_output:
            print(json.dumps(diff, indent=2))
        else:
            print(format_compare(diff, no_color=no_color))
        return 0

    # Inspect mode
    exit_code = 0
    for filepath in args.files:
        if not os.path.exists(filepath):
            print(f"Error: file not found: {filepath}", file=sys.stderr)
            exit_code = 1
            continue

        info = parse_model(filepath)

        # Filter by severity
        sev_order = ["critical", "high", "medium", "low", "info"]
        min_idx = sev_order.index(args.min_severity)
        filtered_findings = [f for f in info.findings if sev_order.index(f.severity) <= min_idx]
        display_info = ModelInfo(
            path=info.path, format=info.format, file_size=info.file_size,
            header_size=info.header_size, tensors=info.tensors,
            metadata=info.metadata, findings=filtered_findings,
            gguf_version=info.gguf_version,
            gguf_tensor_count=info.gguf_tensor_count,
            gguf_kv_count=info.gguf_kv_count,
        )

        stats = compute_stats(info)

        if args.json_output:
            print(format_json_output(display_info, stats))
        else:
            print(format_inspect(display_info, stats, verbose=args.verbose, no_color=no_color))

        # CI mode
        if args.ci:
            grade, _ = calculate_grade(info.findings)
            grades = ["A", "B", "C", "D", "F"]
            if grades.index(grade) > grades.index(args.threshold):
                exit_code = 1

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
