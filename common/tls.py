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
from pathlib import Path
from typing import Iterable, Optional, Tuple

import grpc
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

TLS_DIR_ENV = "GPUFABRIC_TLS_DIR"
CERT_VALIDITY_DAYS = 825
CA_VALIDITY_DAYS = 3650


class TLSConfigError(Exception):
    """Raised when TLS files are missing or unusable."""


def default_tls_dir() -> Path:
    """$GPUFABRIC_TLS_DIR, else $XDG_CONFIG_HOME/gpufabric/tls, else ~/.config/gpufabric/tls."""
    if os.environ.get(TLS_DIR_ENV):
        return Path(os.environ[TLS_DIR_ENV]).expanduser()
    config_home = os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config"
    return Path(config_home) / "gpufabric" / "tls"


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


def init_tls_dir(out_dir: Path, hosts: Iterable[str], force: bool = False) -> None:
    """Create a CA plus server and client certificates in out_dir.

    `hosts` are the worker's IP addresses and/or hostnames, exactly as
    clients will dial them. TLS checks the address against the certificate.
    """
    hosts = [h.strip() for h in hosts if h.strip()]
    if not hosts:
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
        ca_key, ca_cert, hosts[0], ExtendedKeyUsageOID.SERVER_AUTH, hosts
    )
    _write_key(out_dir / "server.key", server_key)
    _write_cert(out_dir / "server.crt", server_cert)

    issue_client_cert(out_dir, "client", out_dir)


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
