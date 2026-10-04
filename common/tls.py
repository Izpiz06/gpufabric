"""Mutual TLS: certificate generation and gRPC credentials.

Layout of a TLS directory (default ~/.config/gpufabric/tls):

    ca.crt       CA certificate, needed by workers and clients
    ca.key       CA private key, only needed to issue more certificates
    server.crt   worker certificate (its IPs/hostnames are in the SAN)
    server.key
    client.crt   client certificate; the worker only accepts clients
    client.key   whose certificate the CA signed
"""

import datetime
import ipaddress
import os
import socket
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import grpc
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

TLS_DIR_ENV = "GPUFABRIC_TLS_DIR"
CERT_VALIDITY_DAYS = 825
CA_VALIDITY_DAYS = 3650

try:
    import ifaddr
except ImportError:
    ifaddr = None

# Interface name prefixes that typically represent virtual/bridge/container networks
VIRTUAL_IFACE_PREFIXES = (
    "docker",
    "br-",
    "veth",
    "virbr",
    "tun",
    "tap",
    "vboxnet",
    "vmnet",
    "flannel",
    "cni",
)


class TLSConfigError(Exception):
    """Raised when TLS files are missing or unusable."""


def default_tls_dir() -> Path:
    """$GPUFABRIC_TLS_DIR, else $XDG_CONFIG_HOME/gpufabric/tls, else ~/.config/gpufabric/tls."""
    if os.environ.get(TLS_DIR_ENV):
        return Path(os.environ[TLS_DIR_ENV]).expanduser()
    config_home = os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config"
    return Path(config_home) / "gpufabric" / "tls"


def detect_host_addresses(
    include_localhost: bool = True,
    include_hostname: bool = True,
) -> List[str]:
    """Automatically detect network addresses and hostnames for worker certificate SANs.

    Detects:
    - Localhost & loopback addresses (127.0.0.1, ::1, localhost)
    - System hostname, FQDN, and .local alias
    - Active primary outbound IPv4 routing address
    - Physical network adapter IPv4/IPv6 addresses (ignoring virtual/docker/bridge adapters)
    """
    candidates: List[str] = []

    if include_localhost:
        candidates.extend(["127.0.0.1", "::1", "localhost"])

    if include_hostname:
        try:
            h = socket.gethostname()
            if h:
                candidates.append(h)
                if "." not in h:
                    candidates.append(f"{h}.local")
        except Exception:
            pass
        try:
            fqdn = socket.getfqdn()
            if fqdn:
                candidates.append(fqdn)
        except Exception:
            pass

    # Primary outbound routing IPv4 address
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            if ip and not ip.startswith("127."):
                candidates.append(ip)
    except Exception:
        pass

    # Inspect network interfaces
    if ifaddr is not None:
        try:
            for adapter in ifaddr.get_adapters():
                name = adapter.name.lower()
                if any(name.startswith(p) for p in VIRTUAL_IFACE_PREFIXES):
                    continue
                for ip_obj in adapter.ips:
                    ip = ip_obj.ip
                    if isinstance(ip, str):
                        if not ip.startswith("127."):
                            candidates.append(ip)
                    elif isinstance(ip, tuple) and len(ip) >= 1:
                        ip_str = ip[0]
                        # Exclude link-local (fe80::) and loopback (::1)
                        if not ip_str.lower().startswith("fe80") and ip_str != "::1":
                            candidates.append(ip_str)
        except Exception:
            pass
    else:
        try:
            _, _, ips = socket.gethostbyname_ex(socket.gethostname())
            for ip in ips:
                if not ip.startswith("127."):
                    candidates.append(ip)
        except Exception:
            pass

    # Deduplicate while preserving order and validating syntax
    seen = set()
    results: List[str] = []
    for host in candidates:
        host = host.strip()
        if not host or host in seen:
            continue
        try:
            ipaddress.ip_address(host)
            seen.add(host)
            results.append(host)
        except ValueError:
            try:
                x509.DNSName(host)
                seen.add(host)
                results.append(host)
            except ValueError:
                pass

    return results


def _new_key() -> ec.EllipticCurvePrivateKey:
    return ec.generate_private_key(ec.SECP256R1())


def _write_key(path: Path, key) -> None:
    data = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    # Create the file owner-only from the start so the key is never world-readable.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    os.chmod(path, 0o600)


def _write_cert(path: Path, cert: x509.Certificate) -> None:
    path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))


def _name(common_name: str) -> x509.Name:
    return x509.Name(
        [
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "GPU Fabric"),
            x509.NameAttribute(NameOID.COMMON_NAME, common_name),
        ]
    )


def _san(hosts: Iterable[str]) -> x509.SubjectAlternativeName:
    entries = []
    for host in hosts:
        try:
            entries.append(x509.IPAddress(ipaddress.ip_address(host)))
        except ValueError:
            entries.append(x509.DNSName(host))
    return x509.SubjectAlternativeName(entries)


def _issue(
    ca_key,
    ca_cert: x509.Certificate,
    common_name: str,
    usage: x509.ObjectIdentifier,
    hosts: Iterable[str] = (),
) -> Tuple[ec.EllipticCurvePrivateKey, x509.Certificate]:
    key = _new_key()
    now = datetime.datetime.now(datetime.timezone.utc)
    builder = (
        x509.CertificateBuilder()
        .subject_name(_name(common_name))
        .issuer_name(ca_cert.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=CERT_VALIDITY_DAYS))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(x509.ExtendedKeyUsage([usage]), critical=False)
    )
    hosts = list(hosts)
    if hosts:
        builder = builder.add_extension(_san(hosts), critical=False)
    return key, builder.sign(ca_key, hashes.SHA256())


def init_tls_dir(
    out_dir: Path,
    hosts: Optional[Iterable[str]] = None,
    force: bool = False,
) -> List[str]:
    """Create a CA plus server and client certificates in out_dir.

    `hosts` are the worker's IP addresses and/or hostnames, exactly as
    clients will dial them. If None, host addresses are automatically detected.
    TLS checks the address against the certificate.

    Returns the list of hostnames and IP addresses included in the server certificate.
    """
    if hosts is None:
        san_hosts = detect_host_addresses()
    else:
        san_hosts = [h.strip() for h in hosts if h.strip()]

    if not san_hosts:
        raise TLSConfigError("At least one worker IP or hostname is required (--hosts)")
    out_dir = Path(out_dir)
    if (out_dir / "ca.key").exists() and not force:
        raise TLSConfigError(f"{out_dir} already has a CA. Use --force to replace it.")
    out_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(out_dir, 0o700)

    ca_key = _new_key()
    now = datetime.datetime.now(datetime.timezone.utc)
    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(_name("GPU Fabric CA"))
        .issuer_name(_name("GPU Fabric CA"))
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=CA_VALIDITY_DAYS))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=False,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .sign(ca_key, hashes.SHA256())
    )
    _write_key(out_dir / "ca.key", ca_key)
    _write_cert(out_dir / "ca.crt", ca_cert)

    server_key, server_cert = _issue(
        ca_key,
        ca_cert,
        san_hosts[0],
        ExtendedKeyUsageOID.SERVER_AUTH,
        san_hosts,
    )
    _write_key(out_dir / "server.key", server_key)
    _write_cert(out_dir / "server.crt", server_cert)

    issue_client_cert(out_dir, "client", out_dir)
    return san_hosts


def issue_client_cert(ca_dir: Path, name: str, out_dir: Path) -> Path:
    """Sign a client certificate with the CA in ca_dir.

    Writes a ready-to-copy bundle to out_dir: ca.crt, client.crt and
    client.key. Copy that directory to the client machine's TLS directory.
    """
    ca_dir, out_dir = Path(ca_dir), Path(out_dir)
    try:
        ca_key = serialization.load_pem_private_key((ca_dir / "ca.key").read_bytes(), None)
        ca_pem = (ca_dir / "ca.crt").read_bytes()
    except FileNotFoundError as e:
        raise TLSConfigError(f"No CA in {ca_dir}. Run `gpufabric-certs init` first.") from e
    ca_cert = x509.load_pem_x509_certificate(ca_pem)
    out_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(out_dir, 0o700)
    key, cert = _issue(ca_key, ca_cert, name, ExtendedKeyUsageOID.CLIENT_AUTH)
    if out_dir.resolve() != ca_dir.resolve():
        (out_dir / "ca.crt").write_bytes(ca_pem)
    _write_key(out_dir / "client.key", key)
    _write_cert(out_dir / "client.crt", cert)
    return out_dir


def _read(tls_dir: Path, *names: str) -> Tuple[bytes, ...]:
    missing = [n for n in names if not (tls_dir / n).is_file()]
    if missing:
        raise TLSConfigError(
            f"Missing {', '.join(missing)} in {tls_dir}. Create certificates with "
            "`gpufabric-certs init --hosts <worker-ip>` and copy them as described in the README."
        )
    return tuple((tls_dir / n).read_bytes() for n in names)


def server_credentials(tls_dir: Path) -> grpc.ServerCredentials:
    """mTLS credentials for a worker: present server.crt, require a CA-signed client cert."""
    ca, cert, key = _read(Path(tls_dir), "ca.crt", "server.crt", "server.key")
    return grpc.ssl_server_credentials(
        [(key, cert)], root_certificates=ca, require_client_auth=True
    )


def channel_credentials(tls_dir: Path) -> grpc.ChannelCredentials:
    """mTLS credentials for a client: trust ca.crt, present client.crt."""
    ca, cert, key = _read(Path(tls_dir), "ca.crt", "client.crt", "client.key")
    return grpc.ssl_channel_credentials(
        root_certificates=ca, private_key=key, certificate_chain=cert
    )


def generate_client_csr(name: str, key: ec.EllipticCurvePrivateKey) -> bytes:
    """Generate a PKCS#10 Certificate Signing Request (CSR) for a client key."""
    csr = (
        x509.CertificateSigningRequestBuilder().subject_name(_name(name)).sign(key, hashes.SHA256())
    )
    return csr.public_bytes(serialization.Encoding.PEM)


def sign_client_csr(
    ca_dir: Path, csr_pem: bytes, name: Optional[str] = None
) -> Tuple[bytes, bytes]:
    """Validate and sign a client CSR using the CA in ca_dir.

    Returns (client_cert_pem, ca_cert_pem).
    """
    ca_dir = Path(ca_dir)
    try:
        ca_key = serialization.load_pem_private_key((ca_dir / "ca.key").read_bytes(), None)
        ca_pem = (ca_dir / "ca.crt").read_bytes()
    except FileNotFoundError as e:
        raise TLSConfigError(f"No CA in {ca_dir}. Run `gpufabric-certs init` first.") from e

    ca_cert = x509.load_pem_x509_certificate(ca_pem)
    try:
        csr = x509.load_pem_x509_csr(csr_pem)
    except Exception as e:
        raise TLSConfigError(f"Malformed Certificate Signing Request: {e}") from e

    if not csr.is_signature_valid:
        raise TLSConfigError("CSR signature verification failed.")

    attrs = csr.subject.get_attributes_for_oid(NameOID.COMMON_NAME)
    common_name = name or (attrs[0].value if attrs else "client")
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(_name(str(common_name)))
        .issuer_name(ca_cert.subject)
        .public_key(csr.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=CERT_VALIDITY_DAYS))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.CLIENT_AUTH]), critical=False)
        .sign(ca_key, hashes.SHA256())
    )

    client_cert_pem = cert.public_bytes(serialization.Encoding.PEM)
    return client_cert_pem, ca_pem


def save_client_bundle(
    tls_dir: Path, client_key_pem: bytes, client_cert_pem: bytes, ca_cert_pem: bytes
) -> None:
    """Save client key, certificate, and CA certificate with secure owner-only permissions."""
    tls_dir = Path(tls_dir)
    tls_dir.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(tls_dir, 0o700)
    except OSError:
        pass

    key_path = tls_dir / "client.key"
    fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(client_key_pem)
    try:
        os.chmod(key_path, 0o600)
    except OSError:
        pass

    cert_path = tls_dir / "client.crt"
    cert_path.write_bytes(client_cert_pem)
    try:
        os.chmod(cert_path, 0o644)
    except OSError:
        pass

    ca_path = tls_dir / "ca.crt"
    ca_path.write_bytes(ca_cert_pem)
    try:
        os.chmod(ca_path, 0o644)
    except OSError:
        pass
