import pickle
import struct
import socket
import time
from typing import Any, Optional 
import logging

from utils.network_metrics import NetworkMetrics

logger = logging.getLogger(__name__)

# Global metrics instances
_network_metrics = NetworkMetrics()

def send_message(
    sock: socket.SocketType,
    message: Any,
    buffer_size_mb: Optional[int] = None,

) -> None:
    """Send a message with optional buffer size configuration and metrics tracking.

    Args:
        sock: Socket to send on
        message: Message to send (will be pickled)
        buffer_size_mb: Buffer size in MB (None = use 4MB default)
        
    """
    start_time = time.time()

    # Set buffer size (device-specific or default)
    buffer_bytes = (
        (buffer_size_mb * 1024 * 1024) if buffer_size_mb else (4 * 1024 * 1024)
    )
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, buffer_bytes)
        logger.info(f"Set send buffer size to {buffer_bytes / (1024 * 1024):.2f} MB")
        
    except OSError:
        
        logger.warning("Unable to set send buffer size, using system default")
        
        pass  # Use system default if unable to set

    data = pickle.dumps(message)
    _network_metrics.record_buffer_size(len(data))
    sock.sendall(struct.pack(">I", len(data)) + data)

    # Record metrics
    duration = time.time() - start_time
    _network_metrics.record_send(len(data), duration)


def receive_message(
    sock: socket.SocketType,
    buffer_size_mb: Optional[int] = None,
) -> Optional[dict]:
    """Receive a message with optional buffer size configuration and metrics tracking.

    Args:
        sock: Socket to receive from
        buffer_size_mb: Buffer size in MB (None = use 4MB default)
       

    Returns:
        Unpickled message or None if socket closed
    """
    start_time = time.time()

    # Set buffer size (device-specific or default)
    buffer_bytes = (
        (buffer_size_mb * 1024 * 1024) if buffer_size_mb else (4 * 1024 * 1024)
    )
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, buffer_bytes)
    except OSError:
        pass  # Use system default if unable to set

    # Read the 4-byte message length header
    raw_msglen = sock.recv(4)
    if not raw_msglen:
        return None

    msglen = struct.unpack(">I", raw_msglen)[0]
    _network_metrics.record_buffer_size(msglen)

    # Read the message data - use smaller chunks for better cross-platform compatibility
    # Chunk size based on buffer size: 1MB for small buffers, up to 4MB for large buffers
    chunk_size_base = min(buffer_bytes // 4, 4 * 1024 * 1024)

    data = b""
    remaining = msglen
    while remaining > 0:
        chunk_size = min(chunk_size_base, remaining)
        chunk = sock.recv(chunk_size)
        if not chunk:
            raise ConnectionError("Socket connection broken while receiving message")
        data += chunk
        remaining -= len(chunk)

    result = pickle.loads(data)

    # Record metrics
    duration = time.time() - start_time
    _network_metrics.record_recv(msglen, duration)

    return result
    