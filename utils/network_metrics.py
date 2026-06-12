"""Network performance metrics collection and logging."""

import logging
import time


def log_network_metrics(metrics: dict, logger: logging.Logger, label: str) -> None:
    """Log a formatted summary of network metrics at INFO level.

    Args:
        metrics: Dict returned by ``NetworkMetrics.get_metrics()``.
        logger: Logger to write to.
        label: Short label appended to the ``[net/<label>]`` prefix.
    """
    if not metrics:
        return
    parts = []
    if "total_send_mb" in metrics:
        parts.append(
            f"sent {metrics['total_send_mb']:.2f} MB"
            f" @ {metrics['send_bandwidth_mbps']:.2f} Mbps"
            f" (avg latency {metrics['avg_send_latency_ms']:.1f} ms)"
        )
    if "total_recv_mb" in metrics:
        parts.append(
            f"recv {metrics['total_recv_mb']:.2f} MB"
            f" @ {metrics['recv_bandwidth_mbps']:.2f} Mbps"
            f" (avg latency {metrics['avg_recv_latency_ms']:.1f} ms)"
        )
    if "avg_buffer_size_kb" in metrics:
        parts.append(
            f"avg buf {metrics['avg_buffer_size_kb']:.1f} KB"
            f" / max {metrics['max_buffer_size_kb']:.1f} KB"
        )
    logger.info(f"[net/{label}] " + " | ".join(parts))


class NetworkMetrics:
    """Track network performance metrics for distributed training."""

    def __init__(self) -> None:
        """Initialise empty metric accumulators."""
        self.send_times = []
        self.recv_times = []
        self.send_bytes = []
        self.recv_bytes = []
        self.buffer_sizes = []
        self.last_log_time = time.time()

    def record_send(self, num_bytes: int, duration: float):
        """Record a send operation."""
        self.send_bytes.append(num_bytes)
        self.send_times.append(duration)

    def record_recv(self, num_bytes: int, duration: float):
        """Record a receive operation."""
        self.recv_bytes.append(num_bytes)
        self.recv_times.append(duration)

    def record_buffer_size(self, size: int):
        """Record current buffer size."""
        self.buffer_sizes.append(size)

    def get_metrics(self, reset: bool = True) -> dict:
        """Get aggregated metrics and optionally reset counters."""
        metrics = {}

        if self.send_bytes:
            total_send_mb = sum(self.send_bytes) / (1024 * 1024)
            total_send_time = sum(self.send_times)
            metrics["send_bandwidth_mbps"] = (
                (total_send_mb * 8) / total_send_time if total_send_time > 0 else 0
            )
            metrics["avg_send_latency_ms"] = (
                sum(self.send_times) / len(self.send_times)
            ) * 1000
            metrics["total_send_mb"] = total_send_mb

        if self.recv_bytes:
            total_recv_mb = sum(self.recv_bytes) / (1024 * 1024)
            total_recv_time = sum(self.recv_times)
            metrics["recv_bandwidth_mbps"] = (
                (total_recv_mb * 8) / total_recv_time if total_recv_time > 0 else 0
            )
            metrics["avg_recv_latency_ms"] = (
                sum(self.recv_times) / len(self.recv_times)
            ) * 1000
            metrics["total_recv_mb"] = total_recv_mb

        if self.buffer_sizes:
            metrics["avg_buffer_size_kb"] = (
                sum(self.buffer_sizes) / len(self.buffer_sizes)
            ) / 1024
            metrics["max_buffer_size_kb"] = max(self.buffer_sizes) / 1024

        if reset:
            self.send_times.clear()
            self.recv_times.clear()
            self.send_bytes.clear()
            self.recv_bytes.clear()
            self.buffer_sizes.clear()
            self.last_log_time = time.time()

        return metrics
