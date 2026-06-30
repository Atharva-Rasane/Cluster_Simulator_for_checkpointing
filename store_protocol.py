#!/usr/bin/env python3

from __future__ import annotations

import json
import socket
import struct
from typing import Any


LENGTH_HEADER = struct.Struct("!I")
CHUNK_SIZE_BYTES = 1024 * 1024


def send_json(connection: socket.socket, message: dict[str, Any]) -> None:
    payload = json.dumps(message, separators=(",", ":")).encode("utf-8")
    connection.sendall(LENGTH_HEADER.pack(len(payload)))
    connection.sendall(payload)


def receive_json(connection: socket.socket) -> dict[str, Any]:
    raw_length = receive_exact(connection, LENGTH_HEADER.size)
    (payload_length,) = LENGTH_HEADER.unpack(raw_length)
    payload = receive_exact(connection, payload_length)
    message = json.loads(payload.decode("utf-8"))
    if not isinstance(message, dict):
        raise ValueError("Protocol message must be a JSON object")
    return message


def send_zero_bytes(connection: socket.socket, total_bytes: int) -> None:
    chunk = b"\0" * CHUNK_SIZE_BYTES
    view = memoryview(chunk)
    remaining = total_bytes

    while remaining > 0:
        amount = min(CHUNK_SIZE_BYTES, remaining)
        connection.sendall(view[:amount])
        remaining -= amount


def send_payload(connection: socket.socket, payload: bytes) -> None:
    view = memoryview(payload)
    offset = 0

    while offset < len(payload):
        end = min(offset + CHUNK_SIZE_BYTES, len(payload))
        connection.sendall(view[offset:end])
        offset = end


def receive_payload(connection: socket.socket, total_bytes: int) -> bytes:
    payload = bytearray(total_bytes)
    view = memoryview(payload)
    offset = 0

    while offset < total_bytes:
        received = connection.recv_into(
            view[offset : offset + min(CHUNK_SIZE_BYTES, total_bytes - offset)]
        )
        if received == 0:
            raise ConnectionError("Connection closed before payload completed")
        offset += received

    return bytes(payload)


def receive_exact(connection: socket.socket, total_bytes: int) -> bytes:
    parts: list[bytes] = []
    remaining = total_bytes

    while remaining > 0:
        data = connection.recv(remaining)
        if not data:
            raise ConnectionError("Connection closed before message completed")
        parts.append(data)
        remaining -= len(data)

    return b"".join(parts)
