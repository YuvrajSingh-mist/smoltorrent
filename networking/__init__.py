"""Networking package — exposes framed TCP send/receive helpers."""

from .send_receive import send_message, receive_message

__all__ = [
    "send_message",
    "receive_message",
]
