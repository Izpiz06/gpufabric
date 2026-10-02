"""Tests for mutual TLS certificates and secure connections."""

import stat

import pytest
from cryptography import x509

from common.certs_cli import main as certs_main
from common.tls import TLSConfigError, init_tls_dir, issue_client_cert


@pytest.fixture(scope="module")
def tls_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("tls")
    init_tls_dir(d, ["127.0.0.1", "localhost"])
    return d


def test_init_creates_files_with_private_keys(tls_dir):
    for name in ("ca", "server", "client"):
        assert (tls_dir / f"{name}.crt").is_file()
        mode = stat.S_IMODE((tls_dir / f"{name}.key").stat().st_mode)
        assert mode == 0o600


def test_server_cert_covers_hosts(tls_dir):
    cert = x509.load_pem_x509_certificate((tls_dir / "server.crt").read_bytes())
    san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    assert [str(ip) for ip in san.get_values_for_type(x509.IPAddress)] == ["127.0.0.1"]
    assert san.get_values_for_type(x509.DNSName) == ["localhost"]


def test_init_refuses_to_overwrite_ca(tls_dir):
    with pytest.raises(TLSConfigError, match="already has a CA"):
        init_tls_dir(tls_dir, ["127.0.0.1"])


def test_init_requires_hosts(tmp_path):
    with pytest.raises(TLSConfigError, match="hostname is required"):
        init_tls_dir(tmp_path, [" "])


def test_certs_cli(tmp_path, capsys):
    assert certs_main(["init", "--hosts", "10.0.0.5,gpu-box", "--dir", str(tmp_path)]) == 0
    assert certs_main(["add-client", "laptop", "--dir", str(tmp_path)]) == 0
    assert (tmp_path / "clients" / "laptop" / "client.key").is_file()
    assert certs_main(["init", "--hosts", "10.0.0.5", "--dir", str(tmp_path)]) == 1
    assert "already has a CA" in capsys.readouterr().err


def test_issue_client_bundle(tls_dir, tmp_path):
    bundle = issue_client_cert(tls_dir, "laptop", tmp_path / "laptop")
    assert sorted(p.name for p in bundle.iterdir()) == ["ca.crt", "client.crt", "client.key"]
    ca = x509.load_pem_x509_certificate((tls_dir / "ca.crt").read_bytes())
    cert = x509.load_pem_x509_certificate((bundle / "client.crt").read_bytes())
    cert.verify_directly_issued_by(ca)
