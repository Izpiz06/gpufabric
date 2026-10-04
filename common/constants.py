# Copyright (c) 2026 Mohammad Izaan
# SPDX-License-Identifier: Apache-2.0

"""Common constants for GPU Fabric."""

DEFAULT_PORT = 50051
API_VERSION = "0.1.0"

# gRPC rejects messages above 4 MiB by default. A 1000x1000 float32 matmul
# request carries two 4 MB matrices, so the default is raised to 256 MiB.
DEFAULT_MAX_MESSAGE_MB = 256
# Protobuf cannot serialize a single message of 2 GiB or more.
MAX_MESSAGE_MB_LIMIT = 2047

# mDNS / DNS-SD LAN service discovery
DISCOVERY_SERVICE_TYPE = "_gpufabric._tcp.local."
DEFAULT_DISCOVERY_TIMEOUT = 3.0
