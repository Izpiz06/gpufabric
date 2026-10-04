"""`gpufabric-certs`: create the certificates for mutual TLS."""

import argparse
import datetime
import sys
from pathlib import Path

from common.constants import DEFAULT_TOKEN_TTL_SECONDS
from common.tls import TLSConfigError, default_tls_dir, init_tls_dir, issue_client_cert
from common.tokens import TokenStore


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

    p_token = sub.add_parser(
        "create-token",
        help="Create a single-use token to enroll a client without SSH/SCP",
    )
    p_token.add_argument(
        "--name",
        type=str,
        default="",
        help="Allowed client name (optional, empty allows any client)",
    )
    p_token.add_argument(
        "--ttl",
        type=int,
        default=DEFAULT_TOKEN_TTL_SECONDS,
        help=f"Token validity in seconds (default: {DEFAULT_TOKEN_TTL_SECONDS})",
    )
    p_token.add_argument("--dir", type=Path, default=None, help="Directory holding CA and tokens")

    p_list = sub.add_parser("list-tokens", help="List enrollment tokens and their statuses")
    p_list.add_argument("--dir", type=Path, default=None, help="Directory holding CA and tokens")

    p_revoke = sub.add_parser("revoke-token", help="Revoke a pending enrollment token")
    p_revoke.add_argument("token", help="The token string to revoke")
    p_revoke.add_argument("--dir", type=Path, default=None, help="Directory holding CA and tokens")

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
            print("For another client machine: gpufabric-certs create-token")
            print("Keep ca.key private: anyone holding it can authorize new clients.")
        elif args.command == "add-client":
            out = issue_client_cert(tls_dir, args.name, tls_dir / "clients" / args.name)
            print(f"Created client bundle in {out}\n")
            print(f"Copy ca.crt, client.crt and client.key from {out} to the client machine's")
            print("~/.config/gpufabric/tls/ (or point GPUFABRIC_TLS_DIR at them).")
        elif args.command == "create-token":
            store = TokenStore(tls_dir / "enrollment_tokens.json")
            tok = store.create_token(client_name=args.name, ttl_seconds=args.ttl)
            exp_utc = datetime.datetime.fromtimestamp(
                tok.expires_at, tz=datetime.timezone.utc
            ).strftime("%Y-%m-%d %H:%M:%S UTC")
            print("Enrollment token generated successfully!\n")
            print(f"Token:       {tok.token}")
            print(f"Client Name: {tok.client_name or '(any client)'}")
            print(f"Expires:     {exp_utc} (in {args.ttl}s)\n")
            print("To enroll a client, run this command on the client machine:")
            print(f"  gpufabric enroll <WORKER_IP> --token {tok.token}")
        elif args.command == "list-tokens":
            store = TokenStore(tls_dir / "enrollment_tokens.json")
            tokens = store.list_tokens()
            if not tokens:
                print("No enrollment tokens found.")
                return 0
            print(f"{'TOKEN':<36} {'CLIENT':<15} {'STATUS':<10} {'EXPIRES AT'}")
            print("-" * 75)
            for t in tokens:
                status = "USED" if t.used else ("EXPIRED" if t.is_expired else "VALID")
                exp_str = datetime.datetime.fromtimestamp(
                    t.expires_at, tz=datetime.timezone.utc
                ).strftime("%Y-%m-%d %H:%M:%S")
                print(f"{t.token:<36} {t.client_name or '*':<15} {status:<10} {exp_str}")
        elif args.command == "revoke-token":
            store = TokenStore(tls_dir / "enrollment_tokens.json")
            if store.revoke_token(args.token):
                print(f"Revoked token: {args.token}")
            else:
                print(f"Token not found: {args.token}", file=sys.stderr)
                return 1
    except TLSConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
