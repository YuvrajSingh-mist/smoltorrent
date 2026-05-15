"""Low-level framed TCP messaging with automatic metrics collection.

Messages are length-prefixed (4-byte big-endian header) and pickled.
A global ``NetworkMetrics`` instance tracks bytes and latency for every send/receive.
"""
import pickle
import struct
import socket
import time
from typing import Any, Optional
import logging

from utils.network_metrics import NetworkMetrics

logger = logging.getLogger(__name__)

_network_metrics = NetworkMetrics()


def send_message(sock: socket.socket, message: Any) -> None:
    """Pickle ``message``, frame it with a 4-byte length header, and send it in full.

    The socket must already be in blocking mode (``settimeout(None)``).

    Args:
        sock: Connected blocking socket.
        message: Any pickleable object to send.
    """
    start_time = time.time()
    data = pickle.dumps(message)
    _network_metrics.record_buffer_size(len(data))
    sock.settimeout(None)
    sock.sendall(struct.pack(">I", len(data)) + data)
    _network_metrics.record_send(len(data), time.time() - start_time)


def receive_message(sock: socket.socket) -> Optional[Any]:
    """Read one framed message from ``sock`` and unpickle it.

    Args:
        sock: Connected blocking socket.

    Returns:
        Unpickled object, or ``None`` if the remote end closed the connection.

    Raises:
        ConnectionError: If the socket closes mid-message.
    """
    start_time = time.time()
    sock.settimeout(None)

    raw_msglen = sock.recv(4)
    if not raw_msglen:
        return None

    msglen = struct.unpack(">I", raw_msglen)[0]
    _network_metrics.record_buffer_size(msglen)

    buf = bytearray(msglen)
    view = memoryview(buf)
    received = 0
    while received < msglen:
        n = sock.recv_into(view[received:], min(65536, msglen - received))
        if not n:
            raise ConnectionError("Socket connection broken while receiving message")
        received += n

    result = pickle.loads(buf)
    _network_metrics.record_recv(msglen, time.time() - start_time)
    return result
