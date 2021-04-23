import yaml
import logging
import socket
from ert_shared.storage.main import bind_socket

logger = logging.getLogger(__name__)


def _get_ip_address() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    return s.getsockname()[0]


def find_open_port(lower: int = 51820, upper: int = 51840) -> int:
    host = _get_ip_address()
    for port in range(lower, upper):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind((host, port))
            sock.close()
            return port
        except socket.error:
            pass
    msg = f"No open port for host {host} in the range {lower}-{upper}"
    logging.exception(msg)
    raise Exception(msg)


class EvaluatorServerConfig:
    def __init__(self, port: int = None) -> None:
        self.host: str = _get_ip_address()
        self.port: int = find_open_port() if port is None else port
        self.socket: socket.socket = bind_socket(self.host, self.port)
        self.url: str = f"ws://{self.host}:{self.port}"
        self.client_uri: str = f"{self.url}/client"
        self.dispatch_uri: str = f"{self.url}/dispatch"

    def get_socket(self) -> socket.socket:
        # The use of _closed seems questionable.
        if self.socket._closed:  # type: ignore
            self.socket = bind_socket(self.host, self.port)
            return self.socket
        return self.socket
