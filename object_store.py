#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import queue
import signal
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from store_protocol import (
    ProtocolError,
    receive_discard,
    receive_json,
    send_json,
    send_zero_bytes,
)


PROTOCOL_VERSION = 1
MIB = 1024 * 1024
_EVENT_STOP = object()


@dataclass(frozen=True)
class CheckpointRecord:
    run_id: str
    experiment: str
    variant: str
    attempt: int
    checkpoint_id: str
    iteration: int
    rank: int
    world_size: int | None
    logical_bytes: int
    wire_bytes: int
    stored_bytes: int
    committed_wall_time: str


@dataclass(frozen=True)
class CheckpointGroupKey:
    run_id: str
    checkpoint_id: str
    iteration: int


@dataclass(frozen=True)
class RankKey:
    run_id: str
    rank: int


@dataclass(frozen=True)
class CheckpointManifest:
    run_id: str
    experiment: str
    variant: str
    attempt: int
    checkpoint_id: str
    iteration: int
    world_size: int
    ranks: tuple[int, ...]
    committed_wall_time: str


@dataclass(frozen=True)
class RequestIdentity:
    request_id: str
    run_id: str
    experiment: str
    variant: str
    attempt: int
    rank: int


@dataclass(frozen=True)
class BandwidthDelay:
    queue_wait_s: float
    service_s: float
    expected_total_s: float
    actual_total_s: float
    concurrent_requests: int
    slowdown_multiplier: float


class AggregateBandwidthResource:
    """
    Shared byte-time scheduler with one aggregate throughput ceiling.

    Concurrent requests reserve time on one storage resource. This prevents
    every request from independently receiving the full configured object-store
    bandwidth.
    """

    def __init__(
        self,
        bandwidth_mb_s: float,
        contention_overhead_percent_per_extra_request: float = 0.0,
    ) -> None:
        if bandwidth_mb_s <= 0:
            raise ValueError("bandwidth_mb_s must be positive")
        if contention_overhead_percent_per_extra_request < 0:
            raise ValueError(
                "contention overhead percentage cannot be negative"
            )

        self.bandwidth_bytes_s = bandwidth_mb_s * MIB
        self.contention_overhead_per_extra_request = (
            contention_overhead_percent_per_extra_request / 100.0
        )
        self._next_available = time.perf_counter()
        self._active_requests = 0
        self._lock = threading.Lock()

    def consume(self, byte_count: int) -> BandwidthDelay:
        if byte_count < 0:
            raise ValueError("byte_count cannot be negative")

        requested_at = time.perf_counter()

        with self._lock:
            self._active_requests += 1
            concurrent_requests = self._active_requests
            slowdown_multiplier = 1.0 + (
                max(0, concurrent_requests - 1)
                * self.contention_overhead_per_extra_request
            )
            service_s = (
                byte_count / self.bandwidth_bytes_s
            ) * slowdown_multiplier
            service_start = max(requested_at, self._next_available)
            service_end = service_start + service_s
            self._next_available = service_end

        queue_wait_s = max(0.0, service_start - requested_at)
        expected_total_s = max(0.0, service_end - requested_at)

        try:
            time.sleep(expected_total_s)
            actual_total_s = time.perf_counter() - requested_at
        finally:
            with self._lock:
                self._active_requests -= 1

        return BandwidthDelay(
            queue_wait_s=queue_wait_s,
            service_s=service_s,
            expected_total_s=expected_total_s,
            actual_total_s=actual_total_s,
            concurrent_requests=concurrent_requests,
            slowdown_multiplier=slowdown_multiplier,
        )


class EventWriter:
    """
    Write JSON-lines events from a dedicated thread.

    Request threads enqueue events rather than repeatedly opening and closing
    the event file inside measured request paths.
    """

    def __init__(self, event_file: Path) -> None:
        self.event_file = event_file
        self.event_file.parent.mkdir(parents=True, exist_ok=True)

        self._queue: queue.Queue[dict[str, Any] | object] = queue.Queue()
        self._thread = threading.Thread(
            target=self._run,
            name="object-store-event-writer",
            daemon=False,
        )
        self._thread.start()

    def append(self, event: dict[str, Any]) -> None:
        copied = dict(event)
        copied["wall_time_utc"] = datetime.now(timezone.utc).isoformat()
        self._queue.put(copied)

    def close(self) -> None:
        self._queue.put(_EVENT_STOP)
        self._thread.join()

    def _run(self) -> None:
        with self.event_file.open(
            "a",
            encoding="utf-8",
            buffering=1,
        ) as file:
            while True:
                item = self._queue.get()

                try:
                    if item is _EVENT_STOP:
                        return

                    file.write(
                        json.dumps(item, separators=(",", ":")) + "\n"
                    )
                finally:
                    self._queue.task_done()


class ObjectStoreServer:
    def __init__(
        self,
        host: str,
        port: int,
        write_bandwidth_mb_s: float,
        read_bandwidth_mb_s: float,
        event_file: Path,
        write_contention_overhead_percent: float = 0.0,
        socket_timeout_s: float = 300.0,
        max_workers: int = 128,
        max_pending_connections: int = 256,
        max_checkpoint_wire_bytes: int = 1024 * 1024 * 1024 * 1024,
    ) -> None:
        if not 1 <= port <= 65535:
            raise ValueError("port must be between 1 and 65535")

        if socket_timeout_s <= 0:
            raise ValueError("socket_timeout_s must be positive")

        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")

        if max_pending_connections < 0:
            raise ValueError(
                "max_pending_connections cannot be negative"
            )

        if max_checkpoint_wire_bytes <= 0:
            raise ValueError(
                "max_checkpoint_wire_bytes must be positive"
            )
        if write_contention_overhead_percent < 0:
            raise ValueError(
                "write_contention_overhead_percent cannot be negative"
            )

        self.host = host
        self.port = port
        self.write_bandwidth_mb_s = write_bandwidth_mb_s
        self.read_bandwidth_mb_s = read_bandwidth_mb_s
        self.write_contention_overhead_percent = (
            write_contention_overhead_percent
        )
        self.event_file = event_file
        self.socket_timeout_s = socket_timeout_s
        self.max_checkpoint_wire_bytes = max_checkpoint_wire_bytes

        self._write_resource = AggregateBandwidthResource(
            write_bandwidth_mb_s,
            contention_overhead_percent_per_extra_request=(
                write_contention_overhead_percent
            ),
        )
        self._read_resource = AggregateBandwidthResource(
            read_bandwidth_mb_s
        )

        self._state_lock = threading.RLock()
        self._log_lock = threading.Lock()

        # Latest globally usable checkpoint for each rank.
        self._latest_by_rank: dict[RankKey, CheckpointRecord] = {}

        # Shards waiting for the rest of their worker group.
        self._pending_groups: dict[
            CheckpointGroupKey,
            dict[int, CheckpointRecord],
        ] = {}

        # Latest complete multi-rank checkpoint for each run.
        self._latest_manifest_by_run: dict[
            str,
            CheckpointManifest,
        ] = {}

        self._event_writer = EventWriter(event_file)

        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="object-store-request",
        )

        # ThreadPoolExecutor has an unbounded internal queue, so this semaphore
        # bounds running plus queued connections.
        self._connection_capacity = threading.BoundedSemaphore(
            max_workers + max_pending_connections
        )

        self._shutdown = threading.Event()
        self._server_start = time.perf_counter()
        self._ready_file: Path | None = None

    def log(
        self,
        operation: str,
        event: str,
        experiment: str = "--",
        message: str = "",
    ) -> None:
        wall_time = datetime.now(timezone.utc).strftime(
            "%H:%M:%S.%f"
        )[:-3]
        elapsed = time.perf_counter() - self._server_start

        line = (
            f"{wall_time} UTC | +{elapsed:09.3f}s | "
            f"host=object-store | exp={experiment:<24} | "
            f"{operation:<18} | {event:<10}"
        )

        if message:
            line += f" | {message}"

        with self._log_lock:
            print(line, flush=True)

    def stop(self) -> None:
        self._shutdown.set()

    def handle_connection(
        self,
        connection: socket.socket,
        address: tuple[str, int],
    ) -> None:
        connection.settimeout(self.socket_timeout_s)

        request_id = "--"
        experiment = "--"

        try:
            request = receive_json(connection)
            operation = self._required_string(request, "op").upper()
            request_id = self._request_id(request)
            experiment = str(request.get("experiment", "--"))

            version = self._optional_integer(
                request,
                "protocol_version",
                default=PROTOCOL_VERSION,
                minimum=1,
            )

            if version != PROTOCOL_VERSION:
                raise ProtocolError(
                    f"Unsupported protocol_version={version}; "
                    f"supported version is {PROTOCOL_VERSION}"
                )

            if operation == "PUT":
                self.handle_put(connection, address, request)

            elif operation == "GET":
                self.handle_get(connection, address, request)

            elif operation == "STATUS":
                self.handle_status(connection, request)

            elif operation == "RESET":
                self.handle_reset(connection, request)

            elif operation == "PING":
                send_json(
                    connection,
                    {
                        "status": "OK",
                        "protocol_version": PROTOCOL_VERSION,
                        "request_id": request_id,
                    },
                )

            else:
                send_json(
                    connection,
                    {
                        "status": "ERROR",
                        "request_id": request_id,
                        "message": (
                            f"Unknown operation: {operation}"
                        ),
                    },
                )

        except (ProtocolError, ValueError, TypeError, KeyError) as error:
            self.log(
                "REQUEST",
                "INVALID",
                experiment,
                (
                    f"peer={address[0]}, "
                    f"request_id={request_id}, error={error}"
                ),
            )
            self._send_error(connection, request_id, str(error))

        except socket.timeout:
            self.log(
                "REQUEST",
                "TIMEOUT",
                experiment,
                (
                    f"peer={address[0]}, "
                    f"request_id={request_id}"
                ),
            )
            self._send_error(
                connection,
                request_id,
                "Socket operation timed out",
            )

        except (
            ConnectionError,
            BrokenPipeError,
            ConnectionResetError,
            OSError,
        ) as error:
            self.log(
                "REQUEST",
                "DISCONNECT",
                experiment,
                (
                    f"peer={address[0]}, "
                    f"request_id={request_id}, error={error}"
                ),
            )

        except Exception as error:
            # The server must continue accepting requests even if one request
            # encounters an unexpected internal failure.
            self.log(
                "REQUEST",
                "ERROR",
                experiment,
                (
                    f"peer={address[0]}, "
                    f"request_id={request_id}, error={error}"
                ),
            )
            self._send_error(
                connection,
                request_id,
                "Internal object-store error",
            )

        finally:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

            connection.close()

    def handle_put(
        self,
        connection: socket.socket,
        address: tuple[str, int],
        request: dict[str, Any],
    ) -> None:
        identity = self._identity(request)

        iteration = self._required_integer(
            request,
            "iteration",
            minimum=0,
        )

        checkpoint_id = self._optional_string(
            request,
            "checkpoint_id",
            default=f"iteration-{iteration}",
        )

        logical_bytes = self._required_integer(
            request,
            "logical_bytes",
            minimum=0,
        )

        wire_bytes = self._required_integer(
            request,
            "wire_bytes",
            minimum=0,
            maximum=self.max_checkpoint_wire_bytes,
        )

        stored_bytes = self._optional_integer(
            request,
            "stored_bytes",
            default=logical_bytes,
            minimum=0,
        )

        world_size = self._optional_world_size(
            request,
            identity.rank,
        )

        receive_start = time.perf_counter()

        self.log(
            "CHECKPOINT/PUT",
            "START",
            identity.experiment,
            (
                f"request_id={identity.request_id}, "
                f"iteration={iteration}, rank={identity.rank}, "
                f"peer={address[0]}, "
                f"logical_mib={logical_bytes / MIB:.2f}, "
                f"wire_mib={wire_bytes / MIB:.2f}, "
                f"stored_mib={stored_bytes / MIB:.2f}"
            ),
        )

        # Bytes travel through Mininet but are not retained in memory.
        receive_discard(connection, wire_bytes)

        network_receive_s = (
            time.perf_counter() - receive_start
        )

        self.log(
            "STORE/NETWORK",
            "RECEIVED",
            identity.experiment,
            (
                f"request_id={identity.request_id}, "
                f"iteration={iteration}, rank={identity.rank}, "
                f"receive_s={network_receive_s:.3f}"
            ),
        )

        self.log(
            "OBJECT/PERSIST",
            "START",
            identity.experiment,
            (
                f"request_id={identity.request_id}, "
                f"iteration={iteration}, rank={identity.rank}, "
                f"aggregate_bandwidth_mib_s="
                f"{self.write_bandwidth_mb_s:.2f}"
            ),
        )

        persist_delay = self._write_resource.consume(
            stored_bytes
        )

        committed_wall_time = datetime.now(
            timezone.utc
        ).isoformat()

        record = CheckpointRecord(
            run_id=identity.run_id,
            experiment=identity.experiment,
            variant=identity.variant,
            attempt=identity.attempt,
            checkpoint_id=checkpoint_id,
            iteration=iteration,
            rank=identity.rank,
            world_size=world_size,
            logical_bytes=logical_bytes,
            wire_bytes=wire_bytes,
            stored_bytes=stored_bytes,
            committed_wall_time=committed_wall_time,
        )

        state = self._stage_or_commit(record)

        total_s = time.perf_counter() - receive_start

        self.log(
            "CHECKPOINT/PUT",
            state["event"],
            identity.experiment,
            (
                f"request_id={identity.request_id}, "
                f"iteration={iteration}, rank={identity.rank}, "
                f"globally_committed="
                f"{state['globally_committed']}, "
                f"queue_wait_s="
                f"{persist_delay.queue_wait_s:.3f}, "
                f"persist_service_s="
                f"{persist_delay.service_s:.3f}, "
                f"concurrent_writers="
                f"{persist_delay.concurrent_requests}, "
                f"slowdown={persist_delay.slowdown_multiplier:.3f}x, "
                f"total_s={total_s:.3f}"
            ),
        )

        self._event_writer.append(
            {
                "type": "checkpoint_put",
                "request_id": identity.request_id,
                "run_id": identity.run_id,
                "experiment": identity.experiment,
                "variant": identity.variant,
                "attempt": identity.attempt,
                "checkpoint_id": checkpoint_id,
                "iteration": iteration,
                "rank": identity.rank,
                "world_size": world_size,
                "logical_bytes": logical_bytes,
                "wire_bytes": wire_bytes,
                "stored_bytes": stored_bytes,
                "network_receive_s": network_receive_s,
                "object_queue_wait_s": (
                    persist_delay.queue_wait_s
                ),
                "object_persist_service_s": (
                    persist_delay.service_s
                ),
                "object_persist_actual_total_s": (
                    persist_delay.actual_total_s
                ),
                "concurrent_writers": (
                    persist_delay.concurrent_requests
                ),
                "contention_slowdown_multiplier": (
                    persist_delay.slowdown_multiplier
                ),
                "total_s": total_s,
                "globally_committed": (
                    state["globally_committed"]
                ),
                "duplicate": state["duplicate"],
                "stale": state["stale"],
                "peer_ip": address[0],
            }
        )

        response_status = (
            "STALE" if state["stale"] else "OK"
        )

        send_json(
            connection,
            {
                "status": response_status,
                "protocol_version": PROTOCOL_VERSION,
                "request_id": identity.request_id,
                "iteration": iteration,
                "rank": identity.rank,
                "checkpoint_id": checkpoint_id,
                "receive_s": network_receive_s,
                "object_queue_wait_s": (
                    persist_delay.queue_wait_s
                ),
                "object_persist_s": (
                    persist_delay.service_s
                ),
                "object_persist_actual_total_s": (
                    persist_delay.actual_total_s
                ),
                "concurrent_writers": (
                    persist_delay.concurrent_requests
                ),
                "contention_slowdown_multiplier": (
                    persist_delay.slowdown_multiplier
                ),
                "total_s": total_s,
                "committed": state["globally_committed"],
                "duplicate": state["duplicate"],
                "stale": state["stale"],
                "latest_iteration": (
                    state["latest_iteration"]
                ),
            },
        )

    def handle_get(
        self,
        connection: socket.socket,
        address: tuple[str, int],
        request: dict[str, Any],
    ) -> None:
        identity = self._identity(request)
        key = RankKey(identity.run_id, identity.rank)

        with self._state_lock:
            record = self._latest_by_rank.get(key)

        if record is None:
            self.log(
                "RECOVERY/GET",
                "MISS",
                identity.experiment,
                (
                    f"request_id={identity.request_id}, "
                    f"rank={identity.rank}, "
                    f"peer={address[0]}"
                ),
            )

            self._event_writer.append(
                {
                    "type": "recovery_get_miss",
                    "request_id": identity.request_id,
                    "run_id": identity.run_id,
                    "experiment": identity.experiment,
                    "variant": identity.variant,
                    "attempt": identity.attempt,
                    "rank": identity.rank,
                    "peer_ip": address[0],
                }
            )

            send_json(
                connection,
                {
                    "status": "NOT_FOUND",
                    "protocol_version": PROTOCOL_VERSION,
                    "request_id": identity.request_id,
                },
            )
            return

        request_start = time.perf_counter()

        self.log(
            "RECOVERY/GET",
            "START",
            identity.experiment,
            (
                f"request_id={identity.request_id}, "
                f"iteration={record.iteration}, "
                f"rank={record.rank}, peer={address[0]}, "
                f"logical_mib="
                f"{record.logical_bytes / MIB:.2f}, "
                f"wire_mib={record.wire_bytes / MIB:.2f}, "
                f"stored_mib="
                f"{record.stored_bytes / MIB:.2f}"
            ),
        )

        read_delay = self._read_resource.consume(
            record.stored_bytes
        )

        send_json(
            connection,
            {
                "status": "OK",
                "protocol_version": PROTOCOL_VERSION,
                "request_id": identity.request_id,
                "run_id": record.run_id,
                "experiment": record.experiment,
                "variant": record.variant,
                "attempt": record.attempt,
                "checkpoint_id": record.checkpoint_id,
                "iteration": record.iteration,
                "rank": record.rank,
                "world_size": record.world_size,
                "logical_bytes": record.logical_bytes,
                "wire_bytes": record.wire_bytes,
                "stored_bytes": record.stored_bytes,
                "object_read_queue_wait_s": (
                    read_delay.queue_wait_s
                ),
                "object_read_s": read_delay.service_s,
                "object_read_actual_total_s": (
                    read_delay.actual_total_s
                ),
                "committed_wall_time": (
                    record.committed_wall_time
                ),
            },
        )

        # Regenerate meaningless checkpoint bytes for the real Mininet
        # transfer instead of storing the original payload in memory.
        send_zero_bytes(connection, record.wire_bytes)

        total_s = time.perf_counter() - request_start

        self.log(
            "RECOVERY/GET",
            "END",
            identity.experiment,
            (
                f"request_id={identity.request_id}, "
                f"iteration={record.iteration}, "
                f"rank={record.rank}, "
                f"queue_wait_s="
                f"{read_delay.queue_wait_s:.3f}, "
                f"read_service_s="
                f"{read_delay.service_s:.3f}, "
                f"total_s={total_s:.3f}"
            ),
        )

        self._event_writer.append(
            {
                "type": "recovery_get",
                "request_id": identity.request_id,
                "run_id": record.run_id,
                "experiment": record.experiment,
                "variant": record.variant,
                "attempt": record.attempt,
                "checkpoint_id": record.checkpoint_id,
                "iteration": record.iteration,
                "rank": record.rank,
                "world_size": record.world_size,
                "logical_bytes": record.logical_bytes,
                "wire_bytes": record.wire_bytes,
                "stored_bytes": record.stored_bytes,
                "object_read_queue_wait_s": (
                    read_delay.queue_wait_s
                ),
                "object_read_service_s": (
                    read_delay.service_s
                ),
                "object_read_actual_total_s": (
                    read_delay.actual_total_s
                ),
                "total_s": total_s,
                "peer_ip": address[0],
            }
        )

    def handle_status(
        self,
        connection: socket.socket,
        request: dict[str, Any],
    ) -> None:
        identity = self._identity(request)
        rank_was_supplied = "rank" in request

        with self._state_lock:
            if rank_was_supplied:
                record = self._latest_by_rank.get(
                    RankKey(
                        identity.run_id,
                        identity.rank,
                    )
                )
                records = (
                    [] if record is None else [record]
                )
            else:
                records = [
                    record
                    for key, record
                    in self._latest_by_rank.items()
                    if key.run_id == identity.run_id
                ]

            manifest = self._latest_manifest_by_run.get(
                identity.run_id
            )

        if not records:
            send_json(
                connection,
                {
                    "status": "NOT_FOUND",
                    "protocol_version": PROTOCOL_VERSION,
                    "request_id": identity.request_id,
                },
            )
            return

        records.sort(key=lambda item: item.rank)
        latest = max(
            records,
            key=lambda item: item.iteration,
        )

        send_json(
            connection,
            {
                "status": "OK",
                "protocol_version": PROTOCOL_VERSION,
                "request_id": identity.request_id,
                "run_id": identity.run_id,
                "iteration": latest.iteration,
                "checkpoint_id": latest.checkpoint_id,
                "logical_bytes": latest.logical_bytes,
                "wire_bytes": latest.wire_bytes,
                "stored_bytes": latest.stored_bytes,
                "committed_wall_time": (
                    latest.committed_wall_time
                ),
                "records": [
                    self._record_summary(record)
                    for record in records
                ],
                "manifest": (
                    None
                    if manifest is None
                    else asdict(manifest)
                ),
            },
        )

    def handle_reset(
        self,
        connection: socket.socket,
        request: dict[str, Any],
    ) -> None:
        request_id = self._request_id(request)
        experiment = self._required_string(
            request,
            "experiment",
        )

        exact_run = any(
            field in request
            for field in (
                "run_id",
                "variant",
                "attempt",
            )
        )

        target_run_id = (
            self._identity(request).run_id
            if exact_run
            else None
        )

        with self._state_lock:
            removed_latest = 0

            for key, record in list(
                self._latest_by_rank.items()
            ):
                matches = (
                    key.run_id == target_run_id
                    if target_run_id is not None
                    else record.experiment == experiment
                )

                if matches:
                    del self._latest_by_rank[key]
                    removed_latest += 1

            removed_pending = 0

            for group_key, shards in list(
                self._pending_groups.items()
            ):
                sample = next(
                    iter(shards.values()),
                    None,
                )

                matches = (
                    group_key.run_id == target_run_id
                    if target_run_id is not None
                    else (
                        sample is not None
                        and sample.experiment == experiment
                    )
                )

                if matches:
                    del self._pending_groups[group_key]
                    removed_pending += len(shards)

            removed_manifests = 0

            for run_id, manifest in list(
                self._latest_manifest_by_run.items()
            ):
                matches = (
                    run_id == target_run_id
                    if target_run_id is not None
                    else manifest.experiment == experiment
                )

                if matches:
                    del self._latest_manifest_by_run[
                        run_id
                    ]
                    removed_manifests += 1

        removed = (
            removed_latest
            + removed_pending
            + removed_manifests
        )

        self.log(
            "STORE/RESET",
            "END",
            experiment,
            (
                f"request_id={request_id}, "
                f"exact_run={exact_run}, removed={removed}"
            ),
        )

        self._event_writer.append(
            {
                "type": "store_reset",
                "request_id": request_id,
                "experiment": experiment,
                "run_id": target_run_id,
                "removed_latest_records": removed_latest,
                "removed_pending_shards": removed_pending,
                "removed_manifests": removed_manifests,
            }
        )

        send_json(
            connection,
            {
                "status": "OK",
                "protocol_version": PROTOCOL_VERSION,
                "request_id": request_id,
                "removed": removed > 0,
                "removed_count": removed,
            },
        )

    def serve_forever(
        self,
        ready_file: Path | None,
    ) -> None:
        self._ready_file = ready_file

        with socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM,
        ) as server:
            server.setsockopt(
                socket.SOL_SOCKET,
                socket.SO_REUSEADDR,
                1,
            )
            server.bind((self.host, self.port))
            server.listen(128)
            server.settimeout(1.0)

            if ready_file is not None:
                ready_file.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                temporary = ready_file.with_suffix(
                    ready_file.suffix + ".tmp"
                )
                temporary.write_text(
                    "ready\n",
                    encoding="utf-8",
                )
                temporary.replace(ready_file)

            self.log(
                "OBJECT_STORE",
                "READY",
                message=(
                    f"listen={self.host}:{self.port}, "
                    f"aggregate_write_mib_s="
                    f"{self.write_bandwidth_mb_s:.2f}, "
                    f"aggregate_read_mib_s="
                    f"{self.read_bandwidth_mb_s:.2f}, "
                    f"socket_timeout_s="
                    f"{self.socket_timeout_s:.1f}"
                ),
            )

            try:
                while not self._shutdown.is_set():
                    acquired = (
                        self._connection_capacity.acquire(
                            timeout=1.0
                        )
                    )

                    if not acquired:
                        continue

                    try:
                        connection, address = (
                            server.accept()
                        )
                    except socket.timeout:
                        self._connection_capacity.release()
                        continue
                    except OSError:
                        self._connection_capacity.release()

                        if self._shutdown.is_set():
                            break

                        raise

                    try:
                        self._executor.submit(
                            self._handle_connection_and_release,
                            connection,
                            address,
                        )
                    except Exception:
                        self._connection_capacity.release()
                        connection.close()
                        raise

            finally:
                self._shutdown.set()

                self._executor.shutdown(
                    wait=True,
                    cancel_futures=False,
                )

                self._event_writer.close()

                if ready_file is not None:
                    try:
                        ready_file.unlink()
                    except FileNotFoundError:
                        pass

                self.log(
                    "OBJECT_STORE",
                    "STOPPED",
                )

    def _handle_connection_and_release(
        self,
        connection: socket.socket,
        address: tuple[str, int],
    ) -> None:
        try:
            self.handle_connection(
                connection,
                address,
            )
        finally:
            self._connection_capacity.release()

    def _stage_or_commit(
        self,
        record: CheckpointRecord,
    ) -> dict[str, Any]:
        rank_key = RankKey(
            record.run_id,
            record.rank,
        )

        with self._state_lock:
            current = self._latest_by_rank.get(rank_key)

            if (
                current is not None
                and record.iteration < current.iteration
            ):
                return {
                    "event": "STALE",
                    "globally_committed": False,
                    "duplicate": False,
                    "stale": True,
                    "latest_iteration": current.iteration,
                }

            # Backward-compatible behavior. If world_size is not supplied,
            # each rank's checkpoint is committed independently.
            if record.world_size is None:
                duplicate = current == record

                if not duplicate:
                    self._latest_by_rank[rank_key] = record

                return {
                    "event": (
                        "DUPLICATE"
                        if duplicate
                        else "COMMIT"
                    ),
                    "globally_committed": True,
                    "duplicate": duplicate,
                    "stale": False,
                    "latest_iteration": record.iteration,
                }

            # When world_size is supplied, wait for every rank before making
            # the checkpoint visible to GET.
            group_key = CheckpointGroupKey(
                record.run_id,
                record.checkpoint_id,
                record.iteration,
            )

            shards = self._pending_groups.setdefault(
                group_key,
                {},
            )

            existing_shard = shards.get(record.rank)

            if existing_shard is not None:
                if existing_shard != record:
                    raise ValueError(
                        "Conflicting duplicate PUT for the "
                        "same run, checkpoint, iteration, "
                        "and rank"
                    )

                duplicate = True

            else:
                self._validate_group_consistency(
                    shards,
                    record,
                )
                shards[record.rank] = record
                duplicate = False

            if len(shards) < record.world_size:
                return {
                    "event": (
                        "DUPLICATE"
                        if duplicate
                        else "STAGED"
                    ),
                    "globally_committed": False,
                    "duplicate": duplicate,
                    "stale": False,
                    "latest_iteration": (
                        current.iteration
                        if current is not None
                        else None
                    ),
                }

            expected_ranks = set(
                range(record.world_size)
            )
            actual_ranks = set(shards)

            if actual_ranks != expected_ranks:
                missing = sorted(
                    expected_ranks - actual_ranks
                )
                extra = sorted(
                    actual_ranks - expected_ranks
                )

                raise ValueError(
                    "Cannot commit checkpoint: "
                    f"missing_ranks={missing}, "
                    f"extra_ranks={extra}"
                )

            committed_time = datetime.now(
                timezone.utc
            ).isoformat()

            committed_records: list[
                CheckpointRecord
            ] = []

            for rank in sorted(shards):
                shard = shards[rank]

                committed = CheckpointRecord(
                    **{
                        **asdict(shard),
                        "committed_wall_time": (
                            committed_time
                        ),
                    }
                )

                existing = self._latest_by_rank.get(
                    RankKey(
                        committed.run_id,
                        committed.rank,
                    )
                )

                if (
                    existing is None
                    or committed.iteration
                    >= existing.iteration
                ):
                    self._latest_by_rank[
                        RankKey(
                            committed.run_id,
                            committed.rank,
                        )
                    ] = committed

                committed_records.append(committed)

            manifest = CheckpointManifest(
                run_id=record.run_id,
                experiment=record.experiment,
                variant=record.variant,
                attempt=record.attempt,
                checkpoint_id=record.checkpoint_id,
                iteration=record.iteration,
                world_size=record.world_size,
                ranks=tuple(sorted(shards)),
                committed_wall_time=committed_time,
            )

            previous_manifest = (
                self._latest_manifest_by_run.get(
                    record.run_id
                )
            )

            if (
                previous_manifest is None
                or manifest.iteration
                >= previous_manifest.iteration
            ):
                self._latest_manifest_by_run[
                    record.run_id
                ] = manifest

            del self._pending_groups[group_key]

            self._discard_older_pending_groups(
                record.run_id,
                record.iteration,
            )

            return {
                "event": "COMMIT",
                "globally_committed": True,
                "duplicate": duplicate,
                "stale": False,
                "latest_iteration": record.iteration,
                "committed_ranks": [
                    item.rank
                    for item in committed_records
                ],
            }

    def _validate_group_consistency(
        self,
        shards: dict[int, CheckpointRecord],
        candidate: CheckpointRecord,
    ) -> None:
        if not shards:
            return

        existing = next(iter(shards.values()))

        fields = (
            "run_id",
            "experiment",
            "variant",
            "attempt",
            "checkpoint_id",
            "iteration",
            "world_size",
        )

        mismatches = [
            field
            for field in fields
            if getattr(existing, field)
            != getattr(candidate, field)
        ]

        if mismatches:
            raise ValueError(
                "Checkpoint shard metadata does not match "
                "the existing group: "
                + ", ".join(mismatches)
            )

    def _discard_older_pending_groups(
        self,
        run_id: str,
        iteration: int,
    ) -> None:
        for group_key in list(
            self._pending_groups
        ):
            if (
                group_key.run_id == run_id
                and group_key.iteration <= iteration
            ):
                del self._pending_groups[group_key]

    def _identity(
        self,
        request: dict[str, Any],
    ) -> RequestIdentity:
        experiment = self._required_string(
            request,
            "experiment",
        )

        variant = self._optional_string(
            request,
            "variant",
            default="default",
        )

        attempt = self._optional_integer(
            request,
            "attempt",
            default=0,
            minimum=0,
        )

        rank = self._optional_integer(
            request,
            "rank",
            default=0,
            minimum=0,
        )

        run_id = self._optional_string(
            request,
            "run_id",
            default=(
                f"{experiment}|{variant}|attempt-{attempt}"
            ),
        )

        return RequestIdentity(
            request_id=self._request_id(request),
            run_id=run_id,
            experiment=experiment,
            variant=variant,
            attempt=attempt,
            rank=rank,
        )

    def _optional_world_size(
        self,
        request: dict[str, Any],
        rank: int,
    ) -> int | None:
        if (
            "world_size" not in request
            or request["world_size"] is None
        ):
            return None

        world_size = self._required_integer(
            request,
            "world_size",
            minimum=1,
        )

        if rank >= world_size:
            raise ValueError(
                f"rank={rank} must be smaller than "
                f"world_size={world_size}"
            )

        return world_size

    def _request_id(
        self,
        request: dict[str, Any],
    ) -> str:
        value = request.get("request_id")

        if value is None:
            operation = str(
                request.get("op", "REQUEST")
            ).upper()
            experiment = str(
                request.get("experiment", "unknown")
            )
            rank = str(request.get("rank", 0))
            iteration = str(
                request.get("iteration", "na")
            )

            return (
                f"{operation}:{experiment}:"
                f"rank-{rank}:iter-{iteration}"
            )

        if (
            not isinstance(value, str)
            or not value.strip()
        ):
            raise ValueError(
                "request_id must be a nonempty string"
            )

        return value.strip()

    def _record_summary(
        self,
        record: CheckpointRecord,
    ) -> dict[str, Any]:
        return {
            "rank": record.rank,
            "iteration": record.iteration,
            "checkpoint_id": record.checkpoint_id,
            "logical_bytes": record.logical_bytes,
            "wire_bytes": record.wire_bytes,
            "stored_bytes": record.stored_bytes,
            "committed_wall_time": (
                record.committed_wall_time
            ),
        }

    def _send_error(
        self,
        connection: socket.socket,
        request_id: str,
        message: str,
    ) -> None:
        try:
            send_json(
                connection,
                {
                    "status": "ERROR",
                    "protocol_version": (
                        PROTOCOL_VERSION
                    ),
                    "request_id": request_id,
                    "message": message,
                },
            )
        except Exception:
            pass

    @staticmethod
    def _required_string(
        request: dict[str, Any],
        field: str,
    ) -> str:
        if field not in request:
            raise ValueError(
                f"Missing required field: {field}"
            )

        value = request[field]

        if (
            not isinstance(value, str)
            or not value.strip()
        ):
            raise ValueError(
                f"{field} must be a nonempty string"
            )

        return value.strip()

    @staticmethod
    def _optional_string(
        request: dict[str, Any],
        field: str,
        default: str,
    ) -> str:
        value = request.get(field, default)

        if (
            not isinstance(value, str)
            or not value.strip()
        ):
            raise ValueError(
                f"{field} must be a nonempty string"
            )

        return value.strip()

    @classmethod
    def _required_integer(
        cls,
        request: dict[str, Any],
        field: str,
        minimum: int | None = None,
        maximum: int | None = None,
    ) -> int:
        if field not in request:
            raise ValueError(
                f"Missing required field: {field}"
            )

        return cls._parse_integer(
            request[field],
            field,
            minimum,
            maximum,
        )

    @classmethod
    def _optional_integer(
        cls,
        request: dict[str, Any],
        field: str,
        default: int,
        minimum: int | None = None,
        maximum: int | None = None,
    ) -> int:
        return cls._parse_integer(
            request.get(field, default),
            field,
            minimum,
            maximum,
        )

    @staticmethod
    def _parse_integer(
        value: Any,
        field: str,
        minimum: int | None,
        maximum: int | None,
    ) -> int:
        if isinstance(value, bool):
            raise ValueError(
                f"{field} must be an integer"
            )

        try:
            parsed = int(value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{field} must be an integer"
            ) from error

        if (
            isinstance(value, float)
            and not value.is_integer()
        ):
            raise ValueError(
                f"{field} must be an integer"
            )

        if (
            minimum is not None
            and parsed < minimum
        ):
            raise ValueError(
                f"{field} must be at least {minimum}"
            )

        if (
            maximum is not None
            and parsed > maximum
        ):
            raise ValueError(
                f"{field} cannot exceed {maximum}"
            )

        return parsed


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--host",
        required=True,
    )

    parser.add_argument(
        "--port",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--write-bandwidth-mb-s",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--read-bandwidth-mb-s",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--write-contention-overhead-percent",
        type=float,
        default=0.0,
        help=(
            "Additional persistence service time percentage "
            "for each concurrent writer beyond the first."
        ),
    )

    parser.add_argument(
        "--event-file",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--ready-file",
        type=Path,
    )

    parser.add_argument(
        "--socket-timeout-s",
        type=float,
        default=300.0,
    )

    parser.add_argument(
        "--max-workers",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--max-pending-connections",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--max-checkpoint-wire-mb",
        type=float,
        default=1024.0 * 1024.0,
        help=(
            "Maximum accepted wire payload in MiB; "
            "default is 1 TiB."
        ),
    )

    args = parser.parse_args()

    if not 1 <= args.port <= 65535:
        parser.error(
            "port must be between 1 and 65535"
        )

    if (
        args.write_bandwidth_mb_s <= 0
        or args.read_bandwidth_mb_s <= 0
    ):
        parser.error(
            "Object-store bandwidths must be positive"
        )

    if args.socket_timeout_s <= 0:
        parser.error(
            "socket-timeout-s must be positive"
        )

    if args.write_contention_overhead_percent < 0:
        parser.error(
            "write-contention-overhead-percent cannot be negative"
        )

    if args.max_workers < 1:
        parser.error(
            "max-workers must be at least 1"
        )

    if args.max_pending_connections < 0:
        parser.error(
            "max-pending-connections cannot be negative"
        )

    if args.max_checkpoint_wire_mb <= 0:
        parser.error(
            "max-checkpoint-wire-mb must be positive"
        )

    return args


def main() -> None:
    args = parse_arguments()

    server = ObjectStoreServer(
        host=args.host,
        port=args.port,
        write_bandwidth_mb_s=(
            args.write_bandwidth_mb_s
        ),
        read_bandwidth_mb_s=(
            args.read_bandwidth_mb_s
        ),
        event_file=args.event_file,
        write_contention_overhead_percent=(
            args.write_contention_overhead_percent
        ),
        socket_timeout_s=args.socket_timeout_s,
        max_workers=args.max_workers,
        max_pending_connections=(
            args.max_pending_connections
        ),
        max_checkpoint_wire_bytes=int(
            args.max_checkpoint_wire_mb * MIB
        ),
    )

    def request_shutdown(
        signum: int,
        _frame: Any,
    ) -> None:
        server.log(
            "OBJECT_STORE",
            "SIGNAL",
            message=f"signal={signum}",
        )
        server.stop()

    signal.signal(
        signal.SIGTERM,
        request_shutdown,
    )
    signal.signal(
        signal.SIGINT,
        request_shutdown,
    )

    server.serve_forever(args.ready_file)


if __name__ == "__main__":
    main()
