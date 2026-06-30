#!/usr/bin/env python3

from __future__ import annotations

import json
import socket
import struct
from typing import Any


LENGTH_HEADER = struct.Struct("!I")
CHUNK_SIZE_BYTES = 1024 * 1024
MAX_JSON_BYTES = 1024 * 1024
_ZERO_CHUNK = bytes(CHUNK_SIZE_BYTES)


class ProtocolError(RuntimeError):
    """Raised when a peer sends an invalid or incomplete protocol message."""


def _validate_byte_count(total_bytes: int, name: str = "total_bytes") -> None:
    if isinstance(total_bytes, bool) or not isinstance(total_bytes, int):
        raise TypeError(f"{name} must be an integer")
    if total_bytes < 0:
        raise ValueError(f"{name} cannot be negative")


def send_json(connection: socket.socket, message: dict[str, Any]) -> None:
    if not isinstance(message, dict):
        raise TypeError("Protocol message must be a dictionary")

    try:
        payload = json.dumps(
            message,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Protocol message is not JSON serializable: {error}"
        ) from error

    if not payload:
        raise ValueError("Protocol message cannot be empty")

    if len(payload) > MAX_JSON_BYTES:
        raise ValueError(
            f"Protocol message is too large: {len(payload)} bytes; "
            f"maximum is {MAX_JSON_BYTES} bytes"
        )

    connection.sendall(LENGTH_HEADER.pack(len(payload)))
    connection.sendall(payload)


def receive_json(
    connection: socket.socket,
    max_payload_bytes: int = MAX_JSON_BYTES,
) -> dict[str, Any]:
    _validate_byte_count(max_payload_bytes, "max_payload_bytes")

    if max_payload_bytes == 0:
        raise ValueError("max_payload_bytes must be positive")

    raw_length = receive_exact(connection, LENGTH_HEADER.size)
    (payload_length,) = LENGTH_HEADER.unpack(raw_length)

    if payload_length == 0:
        raise ProtocolError("JSON payload length cannot be zero")

    if payload_length > max_payload_bytes:
        raise ProtocolError(
            f"JSON payload length {payload_length} exceeds the configured "
            f"maximum of {max_payload_bytes} bytes"
        )

    payload = receive_exact(connection, payload_length)

    try:
        message = json.loads(payload.decode("utf-8"))
    except UnicodeDecodeError as error:
        raise ProtocolError("JSON payload is not valid UTF-8") from error
    except json.JSONDecodeError as error:
        raise ProtocolError(f"Invalid JSON payload: {error}") from error

    if not isinstance(message, dict):
        raise ProtocolError("Protocol message must be a JSON object")

    return message


def send_zero_bytes(connection: socket.socket, total_bytes: int) -> None:
    """
    Send a zero-filled payload without allocating the complete payload.

    This preserves real network traffic through Mininet while keeping memory
    consumption bounded to one chunk.
    """

    _validate_byte_count(total_bytes)

    remaining = total_bytes
    view = memoryview(_ZERO_CHUNK)

    while remaining > 0:
        amount = min(CHUNK_SIZE_BYTES, remaining)
        connection.sendall(view[:amount])
        remaining -= amount


def send_payload(
    connection: socket.socket,
    payload: bytes | bytearray | memoryview,
) -> None:
    """Send an existing payload in bounded chunks."""

    view = memoryview(payload)
    offset = 0

    while offset < len(view):
        end = min(offset + CHUNK_SIZE_BYTES, len(view))
        connection.sendall(view[offset:end])
        offset = end


def receive_payload(
    connection: socket.socket,
    total_bytes: int,
    max_total_bytes: int | None = None,
) -> bytes:
    """
    Receive and retain a complete payload.

    The object-store simulator should normally use receive_discard() because
    simulated checkpoint contents have no meaning. This function remains
    available for protocol users that genuinely require the bytes.
    """

    _validate_byte_count(total_bytes)

    if max_total_bytes is not None:
        _validate_byte_count(max_total_bytes, "max_total_bytes")

        if total_bytes > max_total_bytes:
            raise ProtocolError(
                f"Payload length {total_bytes} exceeds the configured maximum "
                f"of {max_total_bytes} bytes"
            )

    if total_bytes == 0:
        return b""

    payload = bytearray(total_bytes)
    view = memoryview(payload)
    offset = 0

    while offset < total_bytes:
        end = min(offset + CHUNK_SIZE_BYTES, total_bytes)
        received = connection.recv_into(view[offset:end])

        if received == 0:
            raise ConnectionError(
                "Connection closed before payload completed"
            )

        offset += received

    return bytes(payload)


def receive_discard(connection: socket.socket, total_bytes: int) -> None:
    """
    Receive exactly total_bytes while retaining only one bounded buffer.

    The bytes still travel through the real socket and therefore through
    Mininet, but are discarded because their contents have no meaning to the
    comparative simulator.
    """

    _validate_byte_count(total_bytes)

    if total_bytes == 0:
        return

    buffer = bytearray(min(CHUNK_SIZE_BYTES, total_bytes))
    view = memoryview(buffer)
    remaining = total_bytes

    while remaining > 0:
        amount = min(len(buffer), remaining)
        received = connection.recv_into(view[:amount])

        if received == 0:
            raise ConnectionError(
                "Connection closed before payload completed"
            )

        remaining -= received


def receive_exact(connection: socket.socket, total_bytes: int) -> bytes:
    """Receive exactly total_bytes or raise if the peer disconnects."""

    _validate_byte_count(total_bytes)

    if total_bytes == 0:
        return b""

    payload = bytearray(total_bytes)
    view = memoryview(payload)
    offset = 0

    while offset < total_bytes:
        received = connection.recv_into(view[offset:])

        if received == 0:
            raise ConnectionError(
                "Connection closed before message completed"
            )

        offset += received

    return bytes(payload)