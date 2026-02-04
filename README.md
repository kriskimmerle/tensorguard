# tensorguard

**AI Model File Inspector & Security Scanner**

Inspect safetensors and GGUF model files without installing any ML libraries. Parse headers, show tensor metadata, detect security risks, compare model files.

**Zero dependencies.** Python 3.9+ stdlib only.

## Why?

- **No ML libs needed** — Inspect model files without installing PyTorch, TensorFlow, or the HuggingFace ecosystem
- **Security scanning** — Detect exploitable metadata patterns ([Unit42 research](https://unit42.paloaltonetworks.com/rce-vulnerabilities-in-ai-python-libraries/)), suspicious tensor names, file integrity issues
- **Quick inspection** — See what's inside a model file: tensor count, parameter count, dtypes, shapes, metadata
- **Model comparison** — Diff two model files to see added/removed/changed tensors
- **CI-ready** — JSON output, configurable thresholds, exit codes for pipeline integration

## Installation

```bash
# Just copy the single file
curl -O https://raw.githubusercontent.com/kriskimmerle/tensorguard/main/tensorguard.py
chmod +x tensorguard.py

# Or install from source
pip install .
```

## Usage

### Inspect a model file

```bash
# Safetensors
tensorguard model.safetensors

# GGUF
tensorguard model.gguf

# Multiple files
tensorguard *.safetensors

# Verbose (show all tensors and metadata)
tensorguard model.safetensors --verbose
```

Example output:
```
tensorguard v0.1.0 — AI Model File Inspector

FILE
  Path:     model.safetensors
  Format:   safetensors
  Size:     548.2 MB (548,224,000 bytes)
  Header:   12.3 KB (12,288 bytes)
  Tensors:  201
  Metadata: 3 keys

MODEL
  Parameters: 137.00M (137,000,000)

DTYPES
           F16:   200 tensors (136.85M params)
           F32:     1 tensors (0.15M params)

SECURITY
  Grade: A (risk score: 0)
  Findings: 0

  ✅ No security issues detected.
```

### Compare two models

```bash
tensorguard --compare model_v1.safetensors model_v2.safetensors
```

Output:
```
tensorguard v0.1.0 — Model Comparison

  A: model_v1.safetensors (safetensors)
  B: model_v2.safetensors (safetensors)

SUMMARY
  Added:     2
  Removed:   0
  Changed:   3
  Unchanged: 196

ADDED
  + model.new_layer.weight
  + model.new_layer.bias

CHANGED
  ~ model.embed.weight
    dtype: F32 → F16
  ~ model.head.weight
    shape: [768, 50257] → [768, 50304]
```

### JSON output

```bash
tensorguard model.safetensors --json
```

### CI mode

```bash
# Exit 1 if grade below C
tensorguard model.safetensors --ci --threshold C

# Only show high+ severity
tensorguard model.safetensors --min-severity high
```

## Supported Formats

| Format | Extensions | What's Parsed |
|--------|-----------|---------------|
| **Safetensors** | `.safetensors` | Header JSON, tensor metadata, `__metadata__` |
| **GGUF** | `.gguf` | Header, KV metadata, tensor info, architecture |

## Security Rules

| Rule | Severity | Description |
|------|----------|-------------|
| TG01 | Critical | File corruption (truncated, invalid header, too small) |
| TG02 | High | Extremely large header (>100MB, possible payload) |
| TG03 | Medium | Invalid `__metadata__` type |
| TG04 | Medium | Invalid tensor entry format |
| TG05 | High | Overlapping tensor data regions |
| TG06 | High | Tensor extends beyond file boundary |
| TG07 | Medium | Unknown/unrecognized dtype |
| TG08 | Low | Empty tensors (0 elements) |
| TG09 | Low | Gap before tensor data |
| TG10 | Medium/Low | Trailing data after last tensor |
| TG11 | Medium | Underscore-prefixed metadata keys |
| TG12 | Critical | Code execution patterns in metadata (eval, exec, __import__, os.system, trust_remote_code) |
| TG13 | Medium | Base64-encoded payload in metadata |
| TG14 | Low | Unusually large metadata values |
| TG15 | Info | Configuration metadata present |
| TG16 | Medium | Suspicious tensor names (backdoor, trojan, exploit keywords) |
| TG17 | Medium | Extremely long tensor names (possible data embedding) |
| TG18 | Medium | Unusual GGUF version |
| TG19 | High/Medium | Extreme tensor/KV counts |
| TG20 | High | Shell metacharacters in model name |

## Background

### Why inspect model files?

AI model files downloaded from HuggingFace and other registries can contain:

1. **Pickle-based code execution** — `.pt`, `.bin` files using pickle (use [pickleaudit](https://github.com/kriskimmerle/pickleaudit) for these)
2. **Metadata exploitation** — Even "safe" formats like safetensors can have metadata that libraries use to instantiate classes ([Unit42 research, Jan 2026](https://unit42.paloaltonetworks.com/rce-vulnerabilities-in-ai-python-libraries/))
3. **Tampered weights** — [Microsoft research (Feb 2026)](https://www.microsoft.com/en-us/security/blog/2026/02/04/detecting-backdoored-language-models-at-scale/) shows backdoors can be embedded in model weights
4. **File corruption** — Truncated downloads, incomplete transfers

tensorguard focuses on format-level inspection — what the file header says, whether the structure is valid, and whether metadata contains suspicious content. It complements runtime model scanning tools.

### Safetensors format

```
[8 bytes: header_size (u64 LE)]
[header_size bytes: JSON header]
[remaining: raw tensor data]
```

The JSON header maps tensor names to `{dtype, shape, data_offsets}`. An optional `__metadata__` key holds string→string metadata.

### GGUF format

```
[4 bytes: "GGUF" magic]
[4 bytes: version (u32)]
[8 bytes: tensor_count (u64)]
[8 bytes: kv_count (u64)]
[KV pairs...]
[Tensor info...]
[Tensor data...]
```

KV pairs use typed values (strings, ints, floats, arrays). Tensor info includes name, dimensions, GGML type, and data offset.

## Requirements

- Python 3.9+
- Zero external dependencies
- Works on Linux, macOS, Windows

## License

MIT
