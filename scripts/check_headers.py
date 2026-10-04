# Copyright (c) 2026 Mohammad Izaan
# SPDX-License-Identifier: Apache-2.0

"""Verify copyright and license header compliance across source files.

Usage:
    python scripts/check_headers.py
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Directories to scan
SOURCE_DIRS = ["client", "worker", "common", "scripts", "tests", "proto", ".github"]

# Files to exclude (e.g. protoc generated artifacts)
EXCLUDE_FILES = {
    "common/gpufabric_pb2.py",
    "common/gpufabric_pb2.pyi",
    "common/gpufabric_pb2_grpc.py",
}

# Regex patterns for headers
PY_HEADER_RE = re.compile(
    r"^# Copyright \(c\) \d{4} [^\n]+\n# SPDX-License-Identifier: Apache-2\.0",
    re.MULTILINE,
)
PROTO_HEADER_RE = re.compile(
    r"^// Copyright \(c\) \d{4} [^\n]+\n// SPDX-License-Identifier: Apache-2\.0",
    re.MULTILINE,
)


def check_file_header(file_path: Path) -> bool:
    content = file_path.read_text(encoding="utf-8")
    if file_path.suffix in [".py", ".yml", ".yaml"]:
        return bool(PY_HEADER_RE.match(content))
    elif file_path.suffix == ".proto":
        return bool(PROTO_HEADER_RE.match(content))
    return True


def main() -> int:
    missing_or_invalid = []

    for dir_name in SOURCE_DIRS:
        target_dir = ROOT / dir_name
        if not target_dir.exists():
            continue

        for ext in ["*.py", "*.proto", "*.yml", "*.yaml"]:
            for file_path in target_dir.rglob(ext):
                rel_path = file_path.relative_to(ROOT).as_posix()
                if rel_path in EXCLUDE_FILES or "__pycache__" in rel_path:
                    continue

                if not check_file_header(file_path):
                    missing_or_invalid.append(rel_path)

    if missing_or_invalid:
        print("❌ Copyright header missing or invalid in the following files:", file=sys.stderr)
        for path in sorted(missing_or_invalid):
            print(f"  - {path}", file=sys.stderr)
        print("\nExpected header for Python files:", file=sys.stderr)
        print("  # Copyright (c) 2026 <Author Name>", file=sys.stderr)
        print("  # SPDX-License-Identifier: Apache-2.0\n", file=sys.stderr)
        print("Expected header for Proto files:", file=sys.stderr)
        print("  // Copyright (c) 2026 <Author Name>", file=sys.stderr)
        print("  // SPDX-License-Identifier: Apache-2.0", file=sys.stderr)
        return 1

    print("✅ All source files have valid copyright and license headers.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
