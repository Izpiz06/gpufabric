"""Tests for secure client enrollment and token-based pairing."""

import os
import stat
import time
from unittest.mock import MagicMock

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

from client.client import GPUFabricClient
from client.commands import cmd_enroll
from client.enroll import EnrollmentError, enroll_client
from common.certs_cli import main as certs_main
from common.gpufabric_pb2 import EnrollRequest
from common.tls import (
    TLSConfigError,
    generate_client_csr,
    init_tls_dir,
    save_client_bundle,
    sign_client_csr,
)
from common.tokens import TokenStore
from worker.enrollment import EnrollmentServicer, create_enrollment_server
from worker.server import bind, create_grpc_server, load_credentials


@pytest.fixture
def ca_dir(tmp_path):
    d = tmp_path / "worker_tls"
    init_tls_dir(d, ["127.0.0.1", "localhost"])
    return d


@pytest.fixture
def token_store(ca_dir):
    return TokenStore(ca_dir / "enrollment_tokens.json")


# ---------------------------------------------------------------------------
# TokenStore Unit Tests
# ---------------------------------------------------------------------------


def test_token_creation_and_properties(token_store):
    tok = token_store.create_token(client_name="macbook", ttl_seconds=60)
    assert tok.token.startswith("gpf_enroll_")
    assert tok.client_name == "macbook"
    assert not tok.used
    assert not tok.is_expired
    assert tok.is_valid


def test_token_file_permissions_and_persistence(tmp_path):
    store_file = tmp_path / "tokens" / "store.json"
    store = TokenStore(store_file)
    tok = store.create_token(client_name="desktop", ttl_seconds=300)

    # Verify file permissions
    if os.name != "nt":
        file_mode = stat.S_IMODE(store_file.stat().st_mode)
        dir_mode = stat.S_IMODE(store_file.parent.stat().st_mode)
        assert file_mode == 0o600
        assert dir_mode == 0o700

    # Verify persistence across separate store instances
    store2 = TokenStore(store_file)
    tokens = store2.list_tokens()
    assert len(tokens) == 1
    assert tokens[0].token == tok.token
    assert tokens[0].client_name == "desktop"


def test_token_single_use_consumption(token_store):
    tok = token_store.create_token(client_name="laptop", ttl_seconds=100)
    assert token_store.validate_and_consume(tok.token, client_name="laptop") is True
    # Second attempt must fail (single-use)
    assert token_store.validate_and_consume(tok.token, client_name="laptop") is False


def test_token_client_name_matching(token_store):
    tok = token_store.create_token(client_name="workstation", ttl_seconds=100)
    # Mismatched client name rejected
    assert token_store.validate_and_consume(tok.token, client_name="intruder") is False
    # Correct client name succeeds
    assert token_store.validate_and_consume(tok.token, client_name="workstation") is True


def test_token_unnamed_allows_any_client(token_store):
    tok = token_store.create_token(client_name="", ttl_seconds=100)
    assert token_store.validate_and_consume(tok.token, client_name="any-client") is True


def test_token_expiration(token_store):
    tok = token_store.create_token(client_name="temp", ttl_seconds=1)
    # Force expired time
    tok.expires_at = time.time() - 10
    token_store._tokens[tok.token] = tok
    token_store._save()

    assert tok.is_expired is True
    assert tok.is_valid is False
    assert token_store.validate_and_consume(tok.token, client_name="temp") is False


def test_token_revocation_and_cleanup(token_store):
    tok1 = token_store.create_token(client_name="c1", ttl_seconds=100)
    tok2 = token_store.create_token(client_name="c2", ttl_seconds=100)

    assert token_store.revoke_token(tok1.token) is True
    assert token_store.revoke_token("nonexistent") is False
    assert len(token_store.list_tokens()) == 1

    # Expire remaining token and clean
    tok2.expires_at = time.time() - 5
    token_store._tokens[tok2.token] = tok2
    token_store._save()

    removed = token_store.clean_expired()
    assert removed == 1
    assert len(token_store.list_tokens()) == 0


# ---------------------------------------------------------------------------
# CSR and TLS Signing Tests
# ---------------------------------------------------------------------------


def test_generate_and_sign_csr(ca_dir):
    key = ec.generate_private_key(ec.SECP256R1())
    csr_pem = generate_client_csr("test-client", key)

    csr = x509.load_pem_x509_csr(csr_pem)
    assert csr.is_signature_valid is True
    attrs = csr.subject.get_attributes_for_oid(x509.NameOID.COMMON_NAME)
    assert attrs[0].value == "test-client"

    client_cert_pem, ca_cert_pem = sign_client_csr(ca_dir, csr_pem, name="test-client")
    ca_cert = x509.load_pem_x509_certificate(ca_cert_pem)
    client_cert = x509.load_pem_x509_certificate(client_cert_pem)

    # Verify certificate was issued by CA
    client_cert.verify_directly_issued_by(ca_cert)

    # Verify extended key usage includes CLIENT_AUTH
    eku = client_cert.extensions.get_extension_for_class(x509.ExtendedKeyUsage).value
    assert x509.ExtendedKeyUsageOID.CLIENT_AUTH in eku


def test_sign_csr_rejects_corrupted_csr(ca_dir):
    with pytest.raises(TLSConfigError, match="Malformed"):
        sign_client_csr(ca_dir, b"corrupted-pem-data")


def test_save_client_bundle_permissions(tmp_path):
    dest = tmp_path / "saved_client"
    key = ec.generate_private_key(ec.SECP256R1())
    key_pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    save_client_bundle(
        tls_dir=dest,
        client_key_pem=key_pem,
        client_cert_pem=b"CERT DATA",
        ca_cert_pem=b"CA DATA",
    )

    assert (dest / "client.key").is_file()
    assert (dest / "client.crt").is_file()
    assert (dest / "ca.crt").is_file()

    if os.name != "nt":
        assert stat.S_IMODE((dest / "client.key").stat().st_mode) == 0o600
        assert stat.S_IMODE((dest / "client.crt").stat().st_mode) == 0o644
        assert stat.S_IMODE((dest / "ca.crt").stat().st_mode) == 0o644
        assert stat.S_IMODE(dest.stat().st_mode) == 0o700


# ---------------------------------------------------------------------------
# End-to-End Enrollment Server & Client Tests
# ---------------------------------------------------------------------------


@pytest.fixture
def running_worker_and_enroll_server(ca_dir, token_store):
    # Main worker mTLS server on ephemeral port
    worker = create_grpc_server(worker_id="enroll-test-worker", warmup=False)
    worker_port = bind(worker, "127.0.0.1:0", load_credentials(ca_dir, insecure=False))
    worker.start()

    # Standalone enrollment server on ephemeral port
    enroll_server = create_enrollment_server(
        tls_dir=ca_dir,
        host="127.0.0.1",
        port=0,
        token_store=token_store,
    )
    enroll_server.start()
    enroll_port = enroll_server.bound_port

    yield {
        "worker_port": worker_port,
        "enroll_port": enroll_port,
        "token_store": token_store,
        "ca_dir": ca_dir,
    }

    enroll_server.stop(None)
    worker.stop(None)


def test_e2e_successful_enrollment(running_worker_and_enroll_server, tmp_path):
    env = running_worker_and_enroll_server
    token = env["token_store"].create_token(client_name="test-laptop", ttl_seconds=300).token

    client_tls_dir = tmp_path / "enrolled_client"
    dest_dir, info = enroll_client(
        worker_ip="127.0.0.1",
        token=token,
        client_name="test-laptop",
        enroll_port=env["enroll_port"],
        port=env["worker_port"],
        tls_dir=client_tls_dir,
        force=False,
        verify_mtls=True,
    )

    assert dest_dir == client_tls_dir
    assert (client_tls_dir / "client.key").is_file()
    assert (client_tls_dir / "client.crt").is_file()
    assert (client_tls_dir / "ca.crt").is_file()

    # Now verify mTLS operations with newly enrolled certs
    with GPUFabricClient(host="127.0.0.1", port=env["worker_port"], tls_dir=client_tls_dir) as cl:
        health = cl.health()
        assert health.worker_id == "enroll-test-worker"
        assert health.status == "ok"


def test_enrollment_rejected_invalid_token(running_worker_and_enroll_server, tmp_path):
    env = running_worker_and_enroll_server
    client_tls_dir = tmp_path / "client_fail"

    with pytest.raises(EnrollmentError, match="Invalid, expired, or already used"):
        enroll_client(
            worker_ip="127.0.0.1",
            token="invalid_token_12345",
            client_name="laptop",
            enroll_port=env["enroll_port"],
            port=env["worker_port"],
            tls_dir=client_tls_dir,
        )


def test_enrollment_rejected_expired_token(running_worker_and_enroll_server, tmp_path):
    env = running_worker_and_enroll_server
    tok = env["token_store"].create_token(client_name="expired_client", ttl_seconds=1)
    tok.expires_at = time.time() - 10
    env["token_store"]._tokens[tok.token] = tok
    env["token_store"]._save()

    client_tls_dir = tmp_path / "client_expired"
    with pytest.raises(EnrollmentError, match="Invalid, expired, or already used"):
        enroll_client(
            worker_ip="127.0.0.1",
            token=tok.token,
            client_name="expired_client",
            enroll_port=env["enroll_port"],
            port=env["worker_port"],
            tls_dir=client_tls_dir,
        )


def test_enrollment_rejected_token_reuse(running_worker_and_enroll_server, tmp_path):
    env = running_worker_and_enroll_server
    token = env["token_store"].create_token(client_name="reuse_client", ttl_seconds=300).token

    client_tls_1 = tmp_path / "c1"
    enroll_client(
        worker_ip="127.0.0.1",
        token=token,
        client_name="reuse_client",
        enroll_port=env["enroll_port"],
        port=env["worker_port"],
        tls_dir=client_tls_1,
        verify_mtls=False,
    )

    client_tls_2 = tmp_path / "c2"
    with pytest.raises(EnrollmentError, match="Invalid, expired, or already used"):
        enroll_client(
            worker_ip="127.0.0.1",
            token=token,
            client_name="reuse_client",
            enroll_port=env["enroll_port"],
            port=env["worker_port"],
            tls_dir=client_tls_2,
            verify_mtls=False,
        )


def test_enrollment_missing_token_or_csr(ca_dir, token_store):
    servicer = EnrollmentServicer(ca_dir, token_store=token_store)
    resp = servicer.Enroll(EnrollRequest(token="", client_name="", csr_pem=b""), None)
    assert resp.success is False
    assert "Missing enrollment token" in resp.message

    resp2 = servicer.Enroll(EnrollRequest(token="tok", client_name="", csr_pem=b""), None)
    assert resp2.success is False
    assert "Missing Certificate Signing Request" in resp2.message


def test_enrollment_client_name_mismatch(running_worker_and_enroll_server, tmp_path):
    env = running_worker_and_enroll_server
    token = env["token_store"].create_token(client_name="valid_node", ttl_seconds=300).token

    client_tls = tmp_path / "c_mismatch"
    with pytest.raises(EnrollmentError, match="Invalid, expired, or already used"):
        enroll_client(
            worker_ip="127.0.0.1",
            token=token,
            client_name="wrong_node",
            enroll_port=env["enroll_port"],
            port=env["worker_port"],
            tls_dir=client_tls,
            verify_mtls=False,
        )


def test_enrollment_refuses_to_overwrite_without_force(running_worker_and_enroll_server, tmp_path):
    env = running_worker_and_enroll_server
    client_tls = tmp_path / "c_existing"
    client_tls.mkdir(parents=True)
    (client_tls / "client.key").write_text("dummy")
    (client_tls / "client.crt").write_text("dummy")

    with pytest.raises(EnrollmentError, match="Use --force to overwrite"):
        enroll_client(
            worker_ip="127.0.0.1",
            token="dummy_tok",
            client_name="laptop",
            enroll_port=env["enroll_port"],
            port=env["worker_port"],
            tls_dir=client_tls,
        )


# ---------------------------------------------------------------------------
# CLI Commands Unit Tests
# ---------------------------------------------------------------------------


def test_certs_cli_token_commands(ca_dir, capsys):
    # 1. create-token
    assert (
        certs_main(["create-token", "--name", "remote-box", "--ttl", "600", "--dir", str(ca_dir)])
        == 0
    )
    out = capsys.readouterr().out
    assert "Enrollment token generated successfully!" in out
    assert "Token:       gpf_enroll_" in out
    assert "remote-box" in out

    # Extract token
    token = None
    for line in out.splitlines():
        if "Token:" in line:
            token = line.split(":", 1)[1].strip()
    assert token is not None

    # 2. list-tokens
    assert certs_main(["list-tokens", "--dir", str(ca_dir)]) == 0
    list_out = capsys.readouterr().out
    assert token in list_out
    assert "VALID" in list_out
    assert "remote-box" in list_out

    # 3. revoke-token
    assert certs_main(["revoke-token", token, "--dir", str(ca_dir)]) == 0
    revoke_out = capsys.readouterr().out
    assert f"Revoked token: {token}" in revoke_out

    # Verify status changed or token removed
    assert certs_main(["list-tokens", "--dir", str(ca_dir)]) == 0
    list_out2 = capsys.readouterr().out
    assert "No enrollment tokens found" in list_out2 or token not in list_out2


def test_cmd_enroll_cli_execution(running_worker_and_enroll_server, tmp_path, capsys):
    env = running_worker_and_enroll_server
    token = env["token_store"].create_token(client_name="cli-node", ttl_seconds=300).token
    client_tls = tmp_path / "cli_enrolled"

    args = MagicMock()
    args.worker_ip = "127.0.0.1"
    args.token = token
    args.name = "cli-node"
    args.enroll_port = env["enroll_port"]
    args.port = env["worker_port"]
    args.tls_dir = client_tls
    args.force = False
    args.timeout = 5.0

    cmd_enroll(args)
    assert (client_tls / "client.key").is_file()
    assert (client_tls / "client.crt").is_file()
    assert (client_tls / "ca.crt").is_file()
