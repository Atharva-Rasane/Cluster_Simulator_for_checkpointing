#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import socket
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from store_protocol import receive_json, receive_payload, send_json, send_payload


@dataclass(frozen=True)
class CheckpointRecord:
    experiment: str
    iteration: int
    logical_bytes: int
    wire_bytes: int
    payload: bytes
    committed_wall_time: str


class ObjectStoreServer:
    def __init__(
        self,
        host: str,
        port: int,
        write_bandwidth_mb_s: float,
        read_bandwidth_mb_s: float,
        event_file: Path,
    ) -> None:
        self.host = host
        self.port = port
        self.write_bandwidth_mb_s = write_bandwidth_mb_s
        self.read_bandwidth_mb_s = read_bandwidth_mb_s
        self.event_file = event_file

        self._lock = threading.Lock()
        self._event_lock = threading.Lock()
        self._records: dict[str, CheckpointRecord] = {}
        self._server_start = time.perf_counter()

    def log(
        self,
        operation: str,
        event: str,
        experiment: str = "--",
        message: str = "",
    ) -> None:
        wall_time = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]
        elapsed = time.perf_counter() - self._server_start
        line = (
            f"{wall_time} UTC | +{elapsed:09.3f}s | "
            f"host=object-store | exp={experiment:<24} | "
            f"{operation:<18} | {event:<8}"
        )
        if message:
            line += f" | {message}"
        print(line, flush=True)

    def append_event(self, event: dict[str, Any]) -> None:
        event["wall_time_utc"] = datetime.now(timezone.utc).isoformat()
        with self._event_lock:
            self.event_file.parent.mkdir(parents=True, exist_ok=True)
            with self.event_file.open("a", encoding="utf-8") as file:
                file.write(json.dumps(event, separators=(",", ":")) + "\n")

    def handle_connection(self, connection: socket.socket, address: tuple[str, int]) -> None:
        try:
            request = receive_json(connection)
            operation = str(request.get("op", "")).upper()
            experiment = str(request.get("experiment", ""))

            if operation == "PUT":
                self.handle_put(connection, address, request)
            elif operation == "GET":
                self.handle_get(connection, address, experiment)
            elif operation == "STATUS":
                self.handle_status(connection, experiment)
            elif operation == "RESET":
                self.handle_reset(connection, experiment)
            elif operation == "PING":
                send_json(connection, {"status": "OK"})
            else:
                send_json(
                    connection,
                    {"status": "ERROR", "message": f"Unknown operation: {operation}"},
                )
        except Exception as error:  # noqa: BLE001 - server must keep accepting requests
            self.log("REQUEST", "ERROR", message=f"peer={address[0]}, error={error}")
            try:
                send_json(connection, {"status": "ERROR", "message": str(error)})
            except Exception:
                pass
        finally:
            connection.close()

    def handle_put(
        self,
        connection: socket.socket,
        address: tuple[str, int],
        request: dict[str, Any],
    ) -> None:
        experiment = str(request["experiment"])
        iteration = int(request["iteration"])
        logical_bytes = int(request["logical_bytes"])
        wire_bytes = int(request["wire_bytes"])

        receive_start = time.perf_counter()
        self.log(
            "CHECKPOINT/PUT",
            "START",
            experiment,
            (
                f"iteration={iteration}, peer={address[0]}, "
                f"logical_mb={logical_bytes / 1024 / 1024:.2f}, "
                f"wire_mb={wire_bytes / 1024 / 1024:.2f}"
            ),
        )

        payload = receive_payload(connection, wire_bytes)
        receive_duration = time.perf_counter() - receive_start

        self.log(
            "STORE/MEMORY",
            "END",
            experiment,
            f"iteration={iteration}, receive_s={receive_duration:.3f}",
        )

        persist_duration = logical_bytes / 1024 / 1024 / self.write_bandwidth_mb_s
        self.log(
            "OBJECT/PERSIST",
            "START",
            experiment,
            (
                f"iteration={iteration}, bandwidth_mb_s={self.write_bandwidth_mb_s:.2f}, "
                f"expected_s={persist_duration:.3f}"
            ),
        )
        time.sleep(persist_duration)

        record = CheckpointRecord(
            experiment=experiment,
            iteration=iteration,
            logical_bytes=logical_bytes,
            wire_bytes=wire_bytes,
            payload=payload,
            committed_wall_time=datetime.now(timezone.utc).isoformat(),
        )
        with self._lock:
            previous = self._records.get(experiment)
            if previous is None or iteration >= previous.iteration:
                self._records[experiment] = record

        total_duration = time.perf_counter() - receive_start
        self.log(
            "CHECKPOINT/PUT",
            "COMMIT",
            experiment,
            f"iteration={iteration}, duration_s={total_duration:.3f}",
        )

        self.append_event(
            {
                "type": "checkpoint_put",
                "experiment": experiment,
                "iteration": iteration,
                "logical_bytes": logical_bytes,
                "wire_bytes": wire_bytes,
                "network_receive_s": receive_duration,
                "object_persist_s": persist_duration,
                "total_s": total_duration,
                "peer_ip": address[0],
            }
        )

        send_json(
            connection,
            {
                "status": "OK",
                "iteration": iteration,
                "receive_s": receive_duration,
                "object_persist_s": persist_duration,
                "total_s": total_duration,
            },
        )

    def handle_get(
        self,
        connection: socket.socket,
        address: tuple[str, int],
        experiment: str,
    ) -> None:
        with self._lock:
            record = self._records.get(experiment)

        if record is None:
            self.log(
                "RECOVERY/GET",
                "MISS",
                experiment,
                f"peer={address[0]}",
            )
            self.append_event(
                {
                    "type": "recovery_get_miss",
                    "experiment": experiment,
                    "peer_ip": address[0],
                }
            )
            send_json(connection, {"status": "NOT_FOUND"})
            return

        read_delay = record.logical_bytes / 1024 / 1024 / self.read_bandwidth_mb_s
        request_start = time.perf_counter()
        self.log(
            "RECOVERY/GET",
            "START",
            experiment,
            (
                f"iteration={record.iteration}, peer={address[0]}, "
                f"logical_mb={record.logical_bytes / 1024 / 1024:.2f}, "
                f"wire_mb={record.wire_bytes / 1024 / 1024:.2f}, "
                f"object_read_s={read_delay:.3f}"
            ),
        )

        time.sleep(read_delay)
        send_json(
            connection,
            {
                "status": "OK",
                "iteration": record.iteration,
                "logical_bytes": record.logical_bytes,
                "wire_bytes": record.wire_bytes,
                "object_read_s": read_delay,
                "committed_wall_time": record.committed_wall_time,
            },
        )
        send_payload(connection, record.payload)

        total_duration = time.perf_counter() - request_start
        self.log(
            "RECOVERY/GET",
            "END",
            experiment,
            f"iteration={record.iteration}, duration_s={total_duration:.3f}",
        )
        self.append_event(
            {
                "type": "recovery_get",
                "experiment": experiment,
                "iteration": record.iteration,
                "logical_bytes": record.logical_bytes,
                "wire_bytes": record.wire_bytes,
                "object_read_s": read_delay,
                "total_s": total_duration,
                "peer_ip": address[0],
            }
        )

    def handle_status(self, connection: socket.socket, experiment: str) -> None:
        with self._lock:
            record = self._records.get(experiment)

        if record is None:
            send_json(connection, {"status": "NOT_FOUND"})
            return

        send_json(
            connection,
            {
                "status": "OK",
                "iteration": record.iteration,
                "logical_bytes": record.logical_bytes,
                "wire_bytes": record.wire_bytes,
                "committed_wall_time": record.committed_wall_time,
            },
        )

    def handle_reset(self, connection: socket.socket, experiment: str) -> None:
        with self._lock:
            removed = self._records.pop(experiment, None)

        self.log(
            "STORE/RESET",
            "END",
            experiment,
            f"removed={removed is not None}",
        )
        send_json(connection, {"status": "OK", "removed": removed is not None})

    def serve_forever(self, ready_file: Path | None) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.host, self.port))
            server.listen(128)

            if ready_file is not None:
                ready_file.parent.mkdir(parents=True, exist_ok=True)
                ready_file.write_text("ready\n", encoding="utf-8")

            self.log(
                "OBJECT_STORE",
                "READY",
                message=(
                    f"listen={self.host}:{self.port}, "
                    f"write_mb_s={self.write_bandwidth_mb_s:.2f}, "
                    f"read_mb_s={self.read_bandwidth_mb_s:.2f}"
                ),
            )

            while True:
                connection, address = server.accept()
                thread = threading.Thread(
                    target=self.handle_connection,
                    args=(connection, address),
                    daemon=True,
                )
                thread.start()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--write-bandwidth-mb-s", type=float, required=True)
    parser.add_argument("--read-bandwidth-mb-s", type=float, required=True)
    parser.add_argument("--event-file", type=Path, required=True)
    parser.add_argument("--ready-file", type=Path)
    args = parser.parse_args()

    if args.write_bandwidth_mb_s <= 0 or args.read_bandwidth_mb_s <= 0:
        parser.error("Object-store bandwidths must be positive")
    return args


def main() -> None:
    args = parse_arguments()
    server = ObjectStoreServer(
        host=args.host,
        port=args.port,
        write_bandwidth_mb_s=args.write_bandwidth_mb_s,
        read_bandwidth_mb_s=args.read_bandwidth_mb_s,
        event_file=args.event_file,
    )
    server.serve_forever(args.ready_file)


if __name__ == "__main__":
    main()
