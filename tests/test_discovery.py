# Copyright (c) 2026 Mohammad Izaan
# SPDX-License-Identifier: Apache-2.0

"""Tests for LAN service discovery (mDNS / Zeroconf) and worker advertisement."""

from unittest.mock import MagicMock, patch

from client.client import GPUFabricClient
from client.commands import cmd_discover
from common.constants import DEFAULT_PORT, DISCOVERY_SERVICE_TYPE
from common.discovery import (
    DiscoveredCandidate,
    DiscoveredWorker,
    WorkerAdvertiser,
    browse_lan_candidates,
    get_local_ip_addresses,
)
from common.gpufabric_pb2 import HealthResponse


def test_discovered_candidate_primary_address():
    c1 = DiscoveredCandidate(
        name="gpu-box",
        addresses=["127.0.0.1", "192.168.1.50"],
        port=50051,
        properties={"worker_id": "gpu-box", "version": "0.1.0"},
    )
    # Should choose non-loopback
    assert c1.primary_address == "192.168.1.50"
    assert c1.worker_id == "gpu-box"

    c2 = DiscoveredCandidate(
        name="local-box",
        addresses=["127.0.0.1"],
        port=50051,
    )
    assert c2.primary_address == "127.0.0.1"
    assert c2.worker_id == "local-box"


def test_get_local_ip_addresses_explicit():
    ips = get_local_ip_addresses(host="192.168.1.100")
    assert ips == ["192.168.1.100"]

    ips_all = get_local_ip_addresses(host="0.0.0.0")
    assert len(ips_all) > 0


def test_worker_advertiser_lifecycle():
    mock_zc = MagicMock()
    adv = WorkerAdvertiser(
        worker_id="test-worker",
        port=50051,
        host="127.0.0.1",
        properties={"custom_prop": "val"},
        zc=mock_zc,
    )

    with adv:
        assert mock_zc.register_service.called
        info = mock_zc.register_service.call_args[0][0]
        assert info.type == DISCOVERY_SERVICE_TYPE
        assert info.port == 50051
        assert "test-worker" in info.name
        assert info.properties[b"worker_id"] == b"test-worker"

    assert mock_zc.unregister_service.called


def test_browse_lan_candidates_deduplication_and_malformed_props():
    mock_zc = MagicMock()

    # Create 2 mock service infos: one normal, one duplicate address, one with raw bytes
    info1 = MagicMock()
    info1.parsed_scoped_addresses.return_value = ["192.168.1.50"]
    info1.port = 50051
    info1.properties = {b"worker_id": b"node-1", b"version": b"0.1.0"}
    info1.server = "node-1.local."

    info2 = MagicMock()
    info2.parsed_scoped_addresses.return_value = ["192.168.1.50"]
    info2.port = 50051
    info2.properties = {}
    info2.server = "node-1.local."

    info3 = MagicMock()
    info3.parsed_scoped_addresses.return_value = ["192.168.1.51"]
    info3.port = 50052
    info3.properties = {b"worker_id": b"node-2"}
    info3.server = "node-2.local."

    with patch("common.discovery.ServiceBrowser") as mock_browser_cls, patch("time.sleep"):

        def browser_side_effect(zc, service_type, listener):
            listener.found_services = {
                f"node-1.{DISCOVERY_SERVICE_TYPE}": info1,
                f"node-1-dup.{DISCOVERY_SERVICE_TYPE}": info2,
                f"node-2.{DISCOVERY_SERVICE_TYPE}": info3,
            }
            return MagicMock()

        mock_browser_cls.side_effect = browser_side_effect

        candidates = browse_lan_candidates(timeout=0.1, zc=mock_zc)
        # Should deduplicate 192.168.1.50:50051
        assert len(candidates) == 2
        names = {c.name for c in candidates}
        assert "node-1" in names
        assert "node-2" in names


def test_discover_lan_sdk_verified_and_unreachable():
    cand1 = DiscoveredCandidate(
        name="node-online",
        addresses=["192.168.1.50"],
        port=50051,
        properties={"worker_id": "node-online"},
    )
    cand2 = DiscoveredCandidate(
        name="node-offline",
        addresses=["192.168.1.51"],
        port=50051,
        properties={"worker_id": "node-offline"},
    )

    with patch("common.discovery.browse_lan_candidates", return_value=[cand1, cand2]):
        mock_client_online = MagicMock()
        mock_client_online.__enter__.return_value = mock_client_online
        mock_client_online.health.return_value = HealthResponse(
            status="ok",
            worker_id="node-online",
            version="0.1.0",
            gpu_available=True,
        )

        mock_client_offline = MagicMock()
        mock_client_offline.__enter__.return_value = mock_client_offline
        mock_client_offline.health.side_effect = Exception("Connection refused")

        def client_factory(host, port, **kwargs):
            if host == "192.168.1.50":
                return mock_client_online
            return mock_client_offline

        workers = GPUFabricClient.discover_lan(timeout=1.0, _client_factory=client_factory)
        assert len(workers) == 2
        w_online = next(w for w in workers if w.address == "192.168.1.50")
        w_offline = next(w for w in workers if w.address == "192.168.1.51")

        assert w_online.status == "ONLINE"
        assert w_online.gpu_available is True
        assert w_offline.status == "UNREACHABLE"
        assert "Connection refused" in w_offline.error


def test_cmd_discover_explicit_vs_lan(capsys):
    # Test explicit
    mock_client = MagicMock()
    mock_client.target = "192.168.1.50:50051"
    mock_client.discover.return_value = HealthResponse(
        status="ok", worker_id="explicit-worker", version="0.1.0", gpu_available=True
    )
    args = MagicMock()
    args.worker_ip = "192.168.1.50"

    cmd_discover(mock_client, args)
    assert mock_client.discover.called

    # Test LAN automatic
    w = DiscoveredWorker(
        name="auto-worker",
        address="192.168.1.60",
        port=DEFAULT_PORT,
        worker_id="auto-worker",
        status="ONLINE",
        gpu_available=True,
    )
    with patch("client.client.GPUFabricClient.discover_lan", return_value=[w]):
        args_lan = MagicMock()
        args_lan.worker_ip = None
        args_lan.timeout = 1.0
        cmd_discover(None, args_lan, client_kwargs={"insecure": True})
        out = capsys.readouterr().out
        assert "auto-worker" in out or "GPU Fabric Workers" in out
