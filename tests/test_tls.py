"""Tests for mutual TLS certificates and secure connections."""

import os
import stat

import grpc
import pytest
from cryptography import x509

from client.client import GPUFabricClient, GPUFabricError
from common.certs_cli import main as certs_main
from common.gpufabric_pb2 import HealthRequest
from common.gpufabric_pb2_grpc import GPUFabricServiceStub
from common.tls import TLSConfigError, init_tls_dir, issue_client_cert
from worker.server import bind, create_grpc_server, load_credentials


@pytest.fixture(scope="module")
def tls_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("tls")
    init_tls_dir(d, ["127.0.0.1", "localhost"])
    return d


@pytest.fixture(scope="module")
def secure_port(tls_dir):
    server = create_grpc_server(worker_id="tls-worker", warmup=False)
    port = bind(server, "127.0.0.1:0", load_credentials(tls_dir, insecure=False))
    server.start()
    yield port
    server.stop(None)


def test_init_creates_files_with_private_keys(tls_dir):
    for name in ("ca", "server", "client"):
        assert (tls_dir / f"{name}.crt").is_file()
        assert (tls_dir / f"{name}.key").is_file()
        if os.name != "nt":
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


def test_client_with_certs_connects(tls_dir, secure_port):
    with GPUFabricClient(host="127.0.0.1", port=secure_port, tls_dir=tls_dir) as client:
        assert client.health().worker_id == "tls-worker"


def test_added_client_bundle_connects(tls_dir, secure_port, tmp_path):
    bundle = issue_client_cert(tls_dir, "laptop", tmp_path / "laptop")
    assert sorted(p.name for p in bundle.iterdir()) == ["ca.crt", "client.crt", "client.key"]
    with GPUFabricClient(host="localhost", port=secure_port, tls_dir=bundle) as client:
        assert client.health().status == "ok"


def _health(port, credentials):
    channel = grpc.secure_channel(f"127.0.0.1:{port}", credentials)
    try:
        return GPUFabricServiceStub(channel).GetHealth(HealthRequest(), timeout=5)
    finally:
        channel.close()


def test_client_without_certificate_is_rejected(tls_dir, secure_port):
    creds = grpc.ssl_channel_credentials(root_certificates=(tls_dir / "ca.crt").read_bytes())
    with pytest.raises(grpc.RpcError) as exc:
        _health(secure_port, creds)
    assert exc.value.code() == grpc.StatusCode.UNAVAILABLE


def test_client_from_another_ca_is_rejected(tls_dir, secure_port, tmp_path):
    init_tls_dir(tmp_path, ["127.0.0.1"])
    creds = grpc.ssl_channel_credentials(
        root_certificates=(tls_dir / "ca.crt").read_bytes(),
        private_key=(tmp_path / "client.key").read_bytes(),
        certificate_chain=(tmp_path / "client.crt").read_bytes(),
    )
    with pytest.raises(grpc.RpcError) as exc:
        _health(secure_port, creds)
    assert exc.value.code() == grpc.StatusCode.UNAVAILABLE


def test_plaintext_client_is_rejected(secure_port):
    with GPUFabricClient(host="127.0.0.1", port=secure_port, timeout=5, insecure=True) as client:
        with pytest.raises(GPUFabricError) as exc:
            client.health()
    assert exc.value.code == grpc.StatusCode.UNAVAILABLE


def test_client_without_tls_files_explains_setup(tmp_path):
    with pytest.raises(GPUFabricError, match="gpufabric-certs init"):
        GPUFabricClient(host="127.0.0.1", tls_dir=tmp_path)


def test_worker_refuses_to_start_without_certs(tmp_path):
    with pytest.raises(TLSConfigError, match="server.crt"):
        load_credentials(tmp_path, insecure=False)
    assert load_credentials(tmp_path, insecure=True) is None


def test_issue_client_bundle_is_signed_by_ca(tls_dir, tmp_path):
    bundle = issue_client_cert(tls_dir, "desktop", tmp_path / "desktop")
    ca = x509.load_pem_x509_certificate((tls_dir / "ca.crt").read_bytes())
    cert = x509.load_pem_x509_certificate((bundle / "client.crt").read_bytes())
    cert.verify_directly_issued_by(ca)


def test_certs_cli(tmp_path, capsys):
    assert certs_main(["init", "--hosts", "10.0.0.5,gpu-box", "--dir", str(tmp_path)]) == 0
    assert certs_main(["add-client", "laptop", "--dir", str(tmp_path)]) == 0
    assert (tmp_path / "clients" / "laptop" / "client.key").is_file()
    assert certs_main(["init", "--hosts", "10.0.0.5", "--dir", str(tmp_path)]) == 1
    assert "already has a CA" in capsys.readouterr().err
