"""LAN service discovery and advertisement using mDNS / DNS-SD (Zeroconf)."""

import logging
import socket
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from zeroconf import IPVersion, ServiceBrowser, ServiceInfo, ServiceListener, Zeroconf

from common.constants import API_VERSION, DEFAULT_DISCOVERY_TIMEOUT, DISCOVERY_SERVICE_TYPE

logger = logging.getLogger("gpufabric.discovery")


@dataclass
class DiscoveredCandidate:
    """Raw mDNS service advertisement discovered on the LAN."""

    name: str
    addresses: List[str]
    port: int
    properties: Dict[str, str] = field(default_factory=dict)
    server: str = ""

    @property
    def primary_address(self) -> str:
        """Return the first non-loopback address if available, else first address."""
        for addr in self.addresses:
            if not addr.startswith("127.") and addr != "::1":
                return addr
        return self.addresses[0] if self.addresses else "127.0.0.1"

    @property
    def worker_id(self) -> str:
        return self.properties.get("worker_id", self.name)


@dataclass
class DiscoveredWorker:
    """Verified or evaluated GPU Fabric worker."""

    name: str
    address: str
    port: int
    worker_id: str
    version: str = ""
    status: str = "ONLINE"
    gpu_available: bool = False
    error: Optional[str] = None


def get_local_ip_addresses(host: str = "0.0.0.0") -> List[str]:
    """Discover local network IPv4 addresses to advertise."""
    if host not in ("0.0.0.0", "::", ""):
        return [host]

    addresses: List[str] = []
    # Primary outbound routing interface
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            if ip and ip != "127.0.0.1":
                addresses.append(ip)
    except Exception:
        pass

    # Hostname-associated interfaces
    try:
        _, _, ips = socket.gethostbyname_ex(socket.gethostname())
        for ip in ips:
            if ip not in addresses and not ip.startswith("127."):
                addresses.append(ip)
    except Exception:
        pass

    if not addresses:
        addresses = ["127.0.0.1"]
    return addresses


class _DiscoveryListener(ServiceListener):
    def __init__(self):
        self.found_services: Dict[str, ServiceInfo] = {}

    def add_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        info = zc.get_service_info(type_, name)
        if info:
            self.found_services[name] = info

    def update_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        info = zc.get_service_info(type_, name)
        if info:
            self.found_services[name] = info

    def remove_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        self.found_services.pop(name, None)


def browse_lan_candidates(
    timeout: float = DEFAULT_DISCOVERY_TIMEOUT,
    zc: Optional[Zeroconf] = None,
) -> List[DiscoveredCandidate]:
    """Scan local network for GPU Fabric worker advertisements via mDNS."""
    close_zc = False
    if zc is None:
        zc = Zeroconf(ip_version=IPVersion.V4Only)
        close_zc = True

    listener = _DiscoveryListener()
    browser = ServiceBrowser(zc, DISCOVERY_SERVICE_TYPE, listener)

    try:
        # Wait for responses
        time.sleep(max(0.1, timeout))
    finally:
        browser.cancel()
        if close_zc:
            zc.close()

    candidates: List[DiscoveredCandidate] = []
    seen_endpoints = set()

    for service_name, info in listener.found_services.items():
        parsed_addresses = info.parsed_scoped_addresses() or info.parsed_addresses()
        if not parsed_addresses:
            continue

        props: Dict[str, str] = {}
        if info.properties:
            for k, v in info.properties.items():
                k_str = k.decode("utf-8", errors="replace") if isinstance(k, bytes) else str(k)
                v_str = v.decode("utf-8", errors="replace") if isinstance(v, bytes) else str(v)
                props[k_str] = v_str

        # Clean display name from service record name
        display_name = service_name.replace(f".{DISCOVERY_SERVICE_TYPE}", "")

        # Deduplicate endpoints
        key = (parsed_addresses[0], info.port)
        if key in seen_endpoints:
            continue
        seen_endpoints.add(key)

        candidates.append(
            DiscoveredCandidate(
                name=display_name,
                addresses=parsed_addresses,
                port=info.port,
                properties=props,
                server=info.server or "",
            )
        )

    return candidates


class WorkerAdvertiser:
    """Advertises a GPU Fabric worker service on the LAN using mDNS / DNS-SD."""

    def __init__(
        self,
        worker_id: str,
        port: int,
        host: str = "0.0.0.0",
        properties: Optional[Dict[str, Any]] = None,
        zc: Optional[Zeroconf] = None,
    ):
        self.worker_id = worker_id
        self.port = port
        self.host = host
        self.properties = properties or {}
        self._zc = zc
        self._owns_zc = False
        self._service_info: Optional[ServiceInfo] = None
        self._is_registered = False

    def start(self) -> None:
        """Register the service advertisement on the local network."""
        if self._is_registered:
            return

        if self._zc is None:
            self._zc = Zeroconf(ip_version=IPVersion.V4Only)
            self._owns_zc = True

        addresses = get_local_ip_addresses(self.host)
        raw_addresses = []
        for addr in addresses:
            try:
                raw_addresses.append(socket.inet_aton(addr))
            except Exception:
                pass

        if not raw_addresses:
            raw_addresses.append(socket.inet_aton("127.0.0.1"))

        # Clean service name
        safe_name = self.worker_id.replace(".", "-")
        service_name = f"{safe_name}.{DISCOVERY_SERVICE_TYPE}"

        props: Dict[str, str] = {
            "version": API_VERSION,
            "worker_id": self.worker_id,
        }
        for k, v in self.properties.items():
            props[str(k)] = str(v)

        self._service_info = ServiceInfo(
            type_=DISCOVERY_SERVICE_TYPE,
            name=service_name,
            addresses=raw_addresses,
            port=self.port,
            properties=props,
            server=f"{safe_name}.local.",
        )

        try:
            self._zc.register_service(self._service_info)
            self._is_registered = True
            logger.info(
                f"LAN service discovery active: advertised '{service_name}' on port {self.port} ({', '.join(addresses)})"
            )
        except Exception as e:
            logger.warning(f"Failed to advertise worker on LAN via mDNS: {e}")

    def stop(self) -> None:
        """Unregister the service advertisement and cleanup."""
        if self._is_registered and self._service_info and self._zc:
            try:
                self._zc.unregister_service(self._service_info)
            except Exception as e:
                logger.debug(f"Error unregistering service: {e}")
            self._is_registered = False

        if self._owns_zc and self._zc:
            try:
                self._zc.close()
            except Exception as e:
                logger.debug(f"Error closing Zeroconf instance: {e}")
            self._zc = None
            self._owns_zc = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
