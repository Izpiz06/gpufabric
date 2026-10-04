# Copyright (c) 2026 Shreyas Mene
# SPDX-License-Identifier: Apache-2.0

"""`gpufabric-certs`: create the certificates for mutual TLS."""

import argparse
import sys
from pathlib import Path

from common.tls import TLSConfigError, default_tls_dir, init_tls_dir, issue_client_cert


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gpufabric-certs",
        description="Create certificates so workers and clients talk over mutual TLS.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="Create a CA plus worker and client certificates")
    p_init.add_argument(
        "--hosts",
        default=None,
        help="Comma-separated IPs/hostnames clients use to reach the worker, "
        "e.g. 192.168.1.50,gpu-box (default: auto-detect local network addresses)",
    )
    p_init.add_argument("--dir", type=Path, default=None, help="Output directory")
    p_init.add_argument("--force", action="store_true", help="Replace an existing CA")

    p_add = sub.add_parser("add-client", help="Issue a certificate for another client machine")
    p_add.add_argument("name", help="Client name, e.g. laptop")
    p_add.add_argument("--dir", type=Path, default=None, help="Directory holding the CA")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    tls_dir = args.dir or default_tls_dir()
    try:
        if args.command == "init":
            raw_hosts = args.hosts.split(",") if args.hosts is not None else None
            used_hosts = init_tls_dir(tls_dir, raw_hosts, force=args.force)
            if args.hosts is not None:
                print("Using specified worker host addresses for certificate SANs:")
            else:
                print("Automatically detected worker host addresses for certificate SANs:")
            for h in used_hosts:
                print(f"  • {h}")
            print(f"\nCreated CA, worker and client certificates in {tls_dir}\n")
            print("Worker machine: start the worker with this directory (the default).")
            print("This machine can also act as a client right away.")
            print("For another client machine: gpufabric-certs add-client <name>")
            print("Keep ca.key private: anyone holding it can authorize new clients.")
        else:
            out = issue_client_cert(tls_dir, args.name, tls_dir / "clients" / args.name)
            print(f"Created client bundle in {out}\n")
            print(f"Copy ca.crt, client.crt and client.key from {out} to the client machine's")
            print("~/.config/gpufabric/tls/ (or point GPUFABRIC_TLS_DIR at them).")
    except TLSConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
