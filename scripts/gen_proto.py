"""Regenerate protobuf and gRPC Python code from proto/gpufabric.proto.

Usage:
    python scripts/gen_proto.py

The include path maps proto/ to the virtual directory common/, so protoc
emits `from common import gpufabric_pb2` in the gRPC stubs. No manual
import patching is needed after regeneration.
"""

import sys
from pathlib import Path

from grpc_tools import protoc

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    args = [
        "grpc_tools.protoc",
        "-Icommon=proto",
        f"--python_out={ROOT}",
        f"--grpc_python_out={ROOT}",
        f"--pyi_out={ROOT}",
        "common/gpufabric.proto",
    ]
    code = protoc.main(args)
    if code != 0:
        print("protoc failed", file=sys.stderr)
    return code


if __name__ == "__main__":
    sys.exit(main())
