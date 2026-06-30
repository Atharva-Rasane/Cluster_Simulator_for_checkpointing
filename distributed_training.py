#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import random
import socket
import struct
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from store_protocol import receive_json, send_json


CHUNK_SIZE_BYTES = 1024 * 1024
GRADIENT_HEADER = struct.Struct("!IIQ")

PROCESS_FAILURE_EXIT_CODE = 20
NODE_FAILURE_EXIT_CODE = 21


class SimulatedFailure(RuntimeError):
    def __init__(self, failure_type: str, message: str) -> None:
        super().__init__(message)
        self.failure_type = failure_type
        self.exit_code = (
            PROCESS_FAILURE_EXIT_CODE
            if failure_type == "process"
            else NODE_FAILURE_EXIT_CODE
        )


class DistributedTraining:
    """Distributed-training, checkpoint, failure, and recovery procedures."""

    def __init__(
        self,
        experiment: str,
        variant: str,
        attempt: int,
        name: str,
        rank: int,
        world_size: int,
        local_ip: str,
        master_ip: str,
        master_port: int,
        object_store_ip: str,
        object_store_port: int,
        network_bandwidth_mbps: float,
        forward_time: float,
        backward_time: float,
        update_time: float,
        gradient_size_mb: float,
        checkpoint_size_mb: float,
        checkpoint_wire_scale: float,
        dram_capacity_mb: float,
        gpu_to_dram_bandwidth_mb_s: float,
        ssd_capacity_mb: float,
        ssd_bandwidth_mb_s: float,
        process_failure_percent_per_second: float,
        node_failure_percent_per_second: float,
        random_seed: int,
        failure_event_file: Path,
    ) -> None:
        self.experiment = experiment
        self.variant = variant
        self.attempt = attempt
        self.name = name
        self.rank = rank
        self.world_size = world_size

        self.local_ip = local_ip
        self.master_ip = master_ip
        self.master_port = master_port
        self.object_store_ip = object_store_ip
        self.object_store_port = object_store_port
        self.network_bandwidth_mbps = network_bandwidth_mbps

        self.forward_time = forward_time
        self.backward_time = backward_time
        self.update_time = update_time
        self.gradient_size_mb = gradient_size_mb

        self.checkpoint_size_mb = checkpoint_size_mb
        self.checkpoint_wire_scale = checkpoint_wire_scale
        self.dram_capacity_mb = dram_capacity_mb
        self.gpu_to_dram_bandwidth_mb_s = gpu_to_dram_bandwidth_mb_s
        self.ssd_capacity_mb = ssd_capacity_mb
        self.ssd_bandwidth_mb_s = ssd_bandwidth_mb_s

        self.process_failure_rate = process_failure_percent_per_second / 100.0
        self.node_failure_rate = node_failure_percent_per_second / 100.0
        self.random = random.Random(random_seed + attempt * 1_000_003 + rank * 10_007)
        self.failure_event_file = failure_event_file

        self.worker_start_time = time.perf_counter()
        self.current_iteration = 0
        self.current_stage = "startup"
        self.last_completed_iteration = 0
        self.resume_checkpoint_iteration = 0

        self.iteration_metrics: dict[int, dict[str, Any]] = {}
        self.counters: dict[str, int | float] = {
            "gradient_bytes_sent": 0,
            "gradient_bytes_received": 0,
            "checkpoint_count": 0,
            "checkpoint_logical_bytes": 0,
            "checkpoint_wire_bytes_sent": 0,
            "recovery_requests": 0,
            "recovery_hits": 0,
            "recovery_misses": 0,
            "recovery_wire_bytes_received": 0,
            "recovery_total_s": 0.0,
        }

        self._validate_configuration()

    # ------------------------------------------------------------------
    # Validation and logging
    # ------------------------------------------------------------------

    def _validate_configuration(self) -> None:
        if self.world_size < 1:
            raise ValueError("world_size must be at least 1")
        if not 0 <= self.rank < self.world_size:
            raise ValueError("rank must be between 0 and world_size - 1")
        if not 1 <= self.master_port <= 65535:
            raise ValueError("master_port must be between 1 and 65535")
        if not 1 <= self.object_store_port <= 65535:
            raise ValueError("object_store_port must be between 1 and 65535")
        if not 0 < self.checkpoint_wire_scale <= 1:
            raise ValueError("checkpoint_wire_scale must be in (0, 1]")

        positive = {
            "network_bandwidth_mbps": self.network_bandwidth_mbps,
            "gradient_size_mb": self.gradient_size_mb,
            "checkpoint_size_mb": self.checkpoint_size_mb,
            "dram_capacity_mb": self.dram_capacity_mb,
            "gpu_to_dram_bandwidth_mb_s": self.gpu_to_dram_bandwidth_mb_s,
            "ssd_capacity_mb": self.ssd_capacity_mb,
            "ssd_bandwidth_mb_s": self.ssd_bandwidth_mb_s,
        }
        for key, value in positive.items():
            if value <= 0:
                raise ValueError(f"{key} must be positive")

        for key, value in {
            "forward_time": self.forward_time,
            "backward_time": self.backward_time,
            "update_time": self.update_time,
            "process_failure_rate": self.process_failure_rate,
            "node_failure_rate": self.node_failure_rate,
        }.items():
            if value < 0:
                raise ValueError(f"{key} cannot be negative")

        if self.process_failure_rate + self.node_failure_rate >= 1:
            raise ValueError("Combined per-second failure probability must be below 100%")

    def log(
        self,
        iteration: int | None,
        component: str,
        event: str,
        message: str = "",
    ) -> None:
        wall_time = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]
        elapsed = time.perf_counter() - self.worker_start_time
        iteration_text = "--" if iteration is None else f"{iteration:02d}"
        line = (
            f"{wall_time} UTC | +{elapsed:09.3f}s | "
            f"exp={self.experiment:<24} | variant={self.variant:<16} | "
            f"attempt={self.attempt:<2} | host={self.name:<4} | rank={self.rank:<2} | "
            f"iter={iteration_text} | {component:<22} | {event:<9}"
        )
        if message:
            line += f" | {message}"
        print(line, flush=True)

    def log_configuration(
        self,
        target_iterations: int,
        checkpoint_interval: int,
    ) -> None:
        self.log(
            None,
            "WORKER",
            "CONFIG",
            (
                f"world_size={self.world_size}, target_iterations={target_iterations}, "
                f"local_ip={self.local_ip}, master={self.master_ip}:{self.master_port}, "
                f"store={self.object_store_ip}:{self.object_store_port}, "
                f"gradient_mb={self.gradient_size_mb:.2f}, "
                f"checkpoint_logical_mb={self.checkpoint_size_mb:.2f}, "
                f"checkpoint_wire_mb={self.checkpoint_wire_bytes / 1024 / 1024:.2f}, "
                f"checkpoint_interval={checkpoint_interval}, "
                f"process_failure_pct_s={self.process_failure_rate * 100:.4f}, "
                f"node_failure_pct_s={self.node_failure_rate * 100:.4f}"
            ),
        )

    @property
    def checkpoint_logical_bytes(self) -> int:
        return int(self.checkpoint_size_mb * 1024 * 1024)

    @property
    def checkpoint_wire_bytes(self) -> int:
        return max(1, int(self.checkpoint_logical_bytes * self.checkpoint_wire_scale))

    # ------------------------------------------------------------------
    # Failure injection
    # ------------------------------------------------------------------

    def set_stage(self, iteration: int, stage: str) -> None:
        self.current_iteration = iteration
        self.current_stage = stage

    def _failure_probability(self, rate_per_second: float, elapsed_s: float) -> float:
        if rate_per_second <= 0 or elapsed_s <= 0:
            return 0.0
        return 1.0 - math.pow(1.0 - rate_per_second, elapsed_s)

    def maybe_fail(self, elapsed_s: float) -> None:
        process_probability = self._failure_probability(
            self.process_failure_rate, elapsed_s
        )
        node_probability = self._failure_probability(self.node_failure_rate, elapsed_s)
        draw = self.random.random()

        if draw < node_probability:
            raise SimulatedFailure(
                "node",
                f"simulated node failure at iter={self.current_iteration}, stage={self.current_stage}",
            )
        if draw < node_probability + process_probability:
            raise SimulatedFailure(
                "process",
                f"simulated process failure at iter={self.current_iteration}, stage={self.current_stage}",
            )

    def failure_aware_sleep(self, duration_s: float, check_interval_s: float = 0.1) -> None:
        remaining = duration_s
        while remaining > 0:
            step = min(check_interval_s, remaining)
            time.sleep(step)
            self.maybe_fail(step)
            remaining -= step

    def write_failure_event(self, failure: SimulatedFailure) -> None:
        event = {
            "type": "worker_failure",
            "experiment": self.experiment,
            "variant": self.variant,
            "attempt": self.attempt,
            "host": self.name,
            "rank": self.rank,
            "failure_type": failure.failure_type,
            "exit_code": failure.exit_code,
            "iteration": self.current_iteration,
            "stage": self.current_stage,
            "last_completed_iteration": self.last_completed_iteration,
            "resume_checkpoint_iteration": self.resume_checkpoint_iteration,
            "worker_elapsed_s": time.perf_counter() - self.worker_start_time,
            "message": str(failure),
            "wall_time_utc": datetime.now(timezone.utc).isoformat(),
        }
        self.failure_event_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.failure_event_file.with_suffix(".tmp")
        temporary.write_text(json.dumps(event, indent=2), encoding="utf-8")
        temporary.replace(self.failure_event_file)
        self.log(
            self.current_iteration or None,
            "FAILURE",
            failure.failure_type.upper(),
            (
                f"stage={self.current_stage}, last_completed={self.last_completed_iteration}, "
                f"event_file={self.failure_event_file}"
            ),
        )

    # ------------------------------------------------------------------
    # Recovery from external object store
    # ------------------------------------------------------------------

    def recover_from_object_store(self) -> int:
        self.set_stage(0, "recovery/get-checkpoint")
        self.counters["recovery_requests"] = int(self.counters["recovery_requests"]) + 1
        start = time.perf_counter()
        self.log(None, "RECOVERY", "START", "requesting latest object-store checkpoint")

        connection = self._connect(self.object_store_ip, self.object_store_port)
        with connection:
            send_json(
                connection,
                {"op": "GET", "experiment": self.experiment, "rank": self.rank},
            )
            response = receive_json(connection)

            if response.get("status") == "NOT_FOUND":
                duration = time.perf_counter() - start
                self.counters["recovery_misses"] = int(self.counters["recovery_misses"]) + 1
                self.counters["recovery_total_s"] = float(self.counters["recovery_total_s"]) + duration
                self.resume_checkpoint_iteration = 0
                self.log(None, "RECOVERY", "MISS", f"duration_s={duration:.3f}, resume_iteration=1")
                return 0

            if response.get("status") != "OK":
                raise RuntimeError(f"Object-store GET failed: {response}")

            checkpoint_iteration = int(response["iteration"])
            wire_bytes = int(response["wire_bytes"])
            logical_bytes = int(response["logical_bytes"])

            transfer_start = time.perf_counter()
            self.log(
                checkpoint_iteration,
                "RECOVERY/TRANSFER",
                "START",
                (
                    f"logical_mb={logical_bytes / 1024 / 1024:.2f}, "
                    f"wire_mb={wire_bytes / 1024 / 1024:.2f}"
                ),
            )
            self._receive_discard_with_failures(connection, wire_bytes)
            transfer_duration = time.perf_counter() - transfer_start
            total_duration = time.perf_counter() - start

            self.counters["recovery_hits"] = int(self.counters["recovery_hits"]) + 1
            self.counters["recovery_wire_bytes_received"] = (
                int(self.counters["recovery_wire_bytes_received"]) + wire_bytes
            )
            self.counters["recovery_total_s"] = float(self.counters["recovery_total_s"]) + total_duration
            self.resume_checkpoint_iteration = checkpoint_iteration
            self.last_completed_iteration = checkpoint_iteration

            self.log(
                checkpoint_iteration,
                "RECOVERY/TRANSFER",
                "END",
                f"transfer_s={transfer_duration:.3f}, total_s={total_duration:.3f}",
            )
            self.log(
                checkpoint_iteration,
                "RECOVERY",
                "END",
                f"next_iteration={checkpoint_iteration + 1}",
            )
            return checkpoint_iteration

    # ------------------------------------------------------------------
    # Iteration lifecycle
    # ------------------------------------------------------------------

    def _record(self, iteration: int) -> dict[str, Any]:
        return self.iteration_metrics.setdefault(
            iteration,
            {
                "iteration": iteration,
                "forward_s": 0.0,
                "backward_s": 0.0,
                "gradient_sync_s": 0.0,
                "training_s": 0.0,
                "update_s": 0.0,
                "checkpoint_gpu_to_dram_s": 0.0,
                "checkpoint_network_s": 0.0,
                "checkpoint_object_persist_s": 0.0,
                "checkpoint_total_s": 0.0,
                "iteration_s": 0.0,
                "gradient_bytes_sent": 0,
                "gradient_bytes_received": 0,
            },
        )

    def begin_iteration(self, iteration: int, target_iterations: int) -> None:
        self.set_stage(iteration, "iteration/start")
        self._record(iteration)["_start"] = time.perf_counter()
        self.log(iteration, "ITERATION", "START", f"target={target_iterations}")

    def end_iteration(self, iteration: int, target_iterations: int) -> None:
        record = self._record(iteration)
        duration = time.perf_counter() - float(record.pop("_start"))
        record["iteration_s"] = duration
        self.last_completed_iteration = iteration
        self.set_stage(iteration, "iteration/complete")
        self.log(
            iteration,
            "ITERATION",
            "END",
            f"target={target_iterations}, duration_s={duration:.3f}",
        )

    # ------------------------------------------------------------------
    # Training stage
    # ------------------------------------------------------------------

    def forward_pass(self, iteration: int) -> None:
        self.set_stage(iteration, "training/forward")
        start = time.perf_counter()
        self.log(iteration, "TRAIN/FORWARD", "START", f"planned_s={self.forward_time:.3f}")
        self.failure_aware_sleep(self.forward_time)
        duration = time.perf_counter() - start
        self._record(iteration)["forward_s"] = duration
        self.log(iteration, "TRAIN/FORWARD", "END", f"duration_s={duration:.3f}")

    def backward_pass(self, iteration: int) -> None:
        self.set_stage(iteration, "training/backward")
        start = time.perf_counter()
        self.log(iteration, "TRAIN/BACKWARD", "START", f"planned_s={self.backward_time:.3f}")
        self.failure_aware_sleep(self.backward_time)
        duration = time.perf_counter() - start
        self._record(iteration)["backward_s"] = duration
        self.log(iteration, "TRAIN/BACKWARD", "END", f"duration_s={duration:.3f}")

    def gradient_synchronization(self, iteration: int) -> None:
        self.set_stage(iteration, "training/gradient-sync")
        record = self._record(iteration)
        if self.world_size == 1:
            self.log(iteration, "GRADIENT_SYNC", "SKIP", "world_size=1")
            return

        gradient_bytes = int(self.gradient_size_mb * 1024 * 1024)
        start = time.perf_counter()
        self.log(
            iteration,
            "GRADIENT_SYNC",
            "START",
            (
                f"payload_mb={self.gradient_size_mb:.2f}, "
                f"configured_link_mbps={self.network_bandwidth_mbps:.2f}"
            ),
        )

        if self.rank == 0:
            sent, received = self._rank_zero_sync(iteration, gradient_bytes)
        else:
            sent, received = self._nonzero_sync(iteration, gradient_bytes)

        duration = time.perf_counter() - start
        record["gradient_sync_s"] = duration
        record["gradient_bytes_sent"] = sent
        record["gradient_bytes_received"] = received
        self.counters["gradient_bytes_sent"] = int(self.counters["gradient_bytes_sent"]) + sent
        self.counters["gradient_bytes_received"] = int(self.counters["gradient_bytes_received"]) + received
        self.log(
            iteration,
            "GRADIENT_SYNC",
            "END",
            (
                f"duration_s={duration:.3f}, sent_mb={sent / 1024 / 1024:.2f}, "
                f"received_mb={received / 1024 / 1024:.2f}"
            ),
        )

    def _rank_zero_sync(self, iteration: int, gradient_bytes: int) -> tuple[int, int]:
        expected = self.world_size - 1
        connections: dict[int, socket.socket] = {}
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.local_ip, self.master_port))
            server.listen(expected)
            server.settimeout(300)
            self.log(
                iteration,
                "GRADIENT_SYNC",
                "LISTEN",
                f"address={self.local_ip}:{self.master_port}, waiting={expected}",
            )

            while len(connections) < expected:
                connection, address = server.accept()
                header = self._receive_exact_with_failures(connection, GRADIENT_HEADER.size)
                received_iteration, peer_rank, received_bytes = GRADIENT_HEADER.unpack(header)
                if received_iteration != iteration or received_bytes != gradient_bytes:
                    connection.close()
                    raise RuntimeError("Invalid gradient synchronization header")
                connections[peer_rank] = connection
                self.log(
                    iteration,
                    "GRADIENT_SYNC",
                    "CONNECT",
                    f"peer_rank={peer_rank}, peer_ip={address[0]}, connected={len(connections)}/{expected}",
                )

            for peer_rank in sorted(connections):
                self._receive_discard_with_failures(connections[peer_rank], gradient_bytes)
                self.log(iteration, "GRADIENT/GATHER", "END", f"peer_rank={peer_rank}")

            for peer_rank in sorted(connections):
                self._send_zero_with_failures(connections[peer_rank], gradient_bytes)
                connections[peer_rank].close()
                self.log(iteration, "GRADIENT/BCAST", "END", f"peer_rank={peer_rank}")

        return gradient_bytes * expected, gradient_bytes * expected

    def _nonzero_sync(self, iteration: int, gradient_bytes: int) -> tuple[int, int]:
        connection = self._connect(self.master_ip, self.master_port, retry=True)
        with connection:
            connection.sendall(GRADIENT_HEADER.pack(iteration, self.rank, gradient_bytes))
            self._send_zero_with_failures(connection, gradient_bytes)
            self.log(iteration, "GRADIENT/SEND", "END", f"to_rank=0, mb={self.gradient_size_mb:.2f}")
            self._receive_discard_with_failures(connection, gradient_bytes)
            self.log(iteration, "GRADIENT/RECV", "END", f"from_rank=0, mb={self.gradient_size_mb:.2f}")
        return gradient_bytes, gradient_bytes

    def training_stage(self, iteration: int) -> None:
        start = time.perf_counter()
        self.log(iteration, "TRAINING", "START")
        self.forward_pass(iteration)
        self.backward_pass(iteration)
        self.gradient_synchronization(iteration)
        duration = time.perf_counter() - start
        self._record(iteration)["training_s"] = duration
        self.log(iteration, "TRAINING", "END", f"duration_s={duration:.3f}")

    # ------------------------------------------------------------------
    # Update stage
    # ------------------------------------------------------------------

    def update_stage(self, iteration: int) -> None:
        self.set_stage(iteration, "update")
        start = time.perf_counter()
        self.log(iteration, "UPDATE", "START", f"planned_s={self.update_time:.3f}")
        self.failure_aware_sleep(self.update_time)
        duration = time.perf_counter() - start
        self._record(iteration)["update_s"] = duration
        self.log(iteration, "UPDATE", "END", f"duration_s={duration:.3f}")

    # ------------------------------------------------------------------
    # External object-store checkpoint
    # ------------------------------------------------------------------

    def checkpoint_to_object_store(self, iteration: int) -> None:
        if self.checkpoint_size_mb > self.dram_capacity_mb:
            raise RuntimeError("Checkpoint exceeds DRAM staging capacity")

        record = self._record(iteration)
        checkpoint_start = time.perf_counter()
        gpu_to_dram_s = self.checkpoint_size_mb / self.gpu_to_dram_bandwidth_mb_s

        self.set_stage(iteration, "checkpoint/gpu-to-dram")
        self.log(
            iteration,
            "CHECKPOINT/GPU2RAM",
            "START",
            (
                f"logical_mb={self.checkpoint_size_mb:.2f}, "
                f"bandwidth_mb_s={self.gpu_to_dram_bandwidth_mb_s:.2f}, "
                f"expected_s={gpu_to_dram_s:.3f}"
            ),
        )
        phase_start = time.perf_counter()
        self.failure_aware_sleep(gpu_to_dram_s)
        gpu_to_dram_duration = time.perf_counter() - phase_start
        record["checkpoint_gpu_to_dram_s"] = gpu_to_dram_duration
        self.log(iteration, "CHECKPOINT/GPU2RAM", "END", f"duration_s={gpu_to_dram_duration:.3f}")

        self.set_stage(iteration, "checkpoint/send-object-store")
        connection = self._connect(self.object_store_ip, self.object_store_port)
        network_start = time.perf_counter()
        with connection:
            send_json(
                connection,
                {
                    "op": "PUT",
                    "experiment": self.experiment,
                    "iteration": iteration,
                    "logical_bytes": self.checkpoint_logical_bytes,
                    "wire_bytes": self.checkpoint_wire_bytes,
                    "rank": self.rank,
                },
            )
            self.log(
                iteration,
                "CHECKPOINT/NETWORK",
                "START",
                (
                    f"store={self.object_store_ip}:{self.object_store_port}, "
                    f"wire_mb={self.checkpoint_wire_bytes / 1024 / 1024:.2f}"
                ),
            )
            self._send_zero_with_failures(connection, self.checkpoint_wire_bytes)
            network_duration = time.perf_counter() - network_start
            self.log(iteration, "CHECKPOINT/NETWORK", "SENT", f"duration_s={network_duration:.3f}")

            self.set_stage(iteration, "checkpoint/object-persist")
            response = receive_json(connection)
            if response.get("status") != "OK":
                raise RuntimeError(f"Object-store PUT failed: {response}")

        total_duration = time.perf_counter() - checkpoint_start
        object_persist_s = float(response.get("object_persist_s", 0.0))
        record["checkpoint_network_s"] = network_duration
        record["checkpoint_object_persist_s"] = object_persist_s
        record["checkpoint_total_s"] = total_duration

        self.counters["checkpoint_count"] = int(self.counters["checkpoint_count"]) + 1
        self.counters["checkpoint_logical_bytes"] = (
            int(self.counters["checkpoint_logical_bytes"]) + self.checkpoint_logical_bytes
        )
        self.counters["checkpoint_wire_bytes_sent"] = (
            int(self.counters["checkpoint_wire_bytes_sent"]) + self.checkpoint_wire_bytes
        )

        self.log(
            iteration,
            "CHECKPOINT/OBJECT",
            "COMMIT",
            (
                f"total_s={total_duration:.3f}, network_s={network_duration:.3f}, "
                f"object_persist_s={object_persist_s:.3f}"
            ),
        )

    # ------------------------------------------------------------------
    # Kept local checkpoint paths; not selected by current workers
    # ------------------------------------------------------------------

    def checkpoint_to_dram(self, iteration: int) -> None:
        if self.checkpoint_size_mb > self.dram_capacity_mb:
            raise RuntimeError("Checkpoint exceeds DRAM capacity")
        self.set_stage(iteration, "checkpoint/local-dram")
        duration = self.checkpoint_size_mb / self.gpu_to_dram_bandwidth_mb_s
        self.log(iteration, "CHECKPOINT/DRAM", "START", f"expected_s={duration:.3f}")
        self.failure_aware_sleep(duration)
        self.log(iteration, "CHECKPOINT/DRAM", "END", f"duration_s={duration:.3f}")

    def checkpoint_to_ssd(self, iteration: int) -> None:
        if self.checkpoint_size_mb > self.ssd_capacity_mb:
            raise RuntimeError("Checkpoint exceeds SSD capacity")
        self.checkpoint_to_dram(iteration)
        self.set_stage(iteration, "checkpoint/local-ssd")
        duration = self.checkpoint_size_mb / self.ssd_bandwidth_mb_s
        self.log(iteration, "CHECKPOINT/SSD", "START", f"expected_s={duration:.3f}")
        self.failure_aware_sleep(duration)
        self.log(iteration, "CHECKPOINT/SSD", "END", f"duration_s={duration:.3f}")

    # ------------------------------------------------------------------
    # Socket helpers
    # ------------------------------------------------------------------

    def _connect(
        self,
        host: str,
        port: int,
        retry: bool = False,
        timeout_s: float = 300.0,
    ) -> socket.socket:
        deadline = time.monotonic() + timeout_s
        while True:
            connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            connection.settimeout(5.0)
            try:
                connection.connect((host, port))
                connection.settimeout(None)
                return connection
            except OSError as error:
                connection.close()
                if not retry or time.monotonic() >= deadline:
                    raise ConnectionError(f"Could not connect to {host}:{port}") from error
                self.maybe_fail(0.05)
                time.sleep(0.05)

    def _send_zero_with_failures(self, connection: socket.socket, total_bytes: int) -> None:
        chunk = b"\0" * CHUNK_SIZE_BYTES
        view = memoryview(chunk)
        remaining = total_bytes
        last_check = time.perf_counter()
        while remaining > 0:
            amount = min(CHUNK_SIZE_BYTES, remaining)
            connection.sendall(view[:amount])
            remaining -= amount
            now = time.perf_counter()
            self.maybe_fail(now - last_check)
            last_check = now

    def _receive_discard_with_failures(self, connection: socket.socket, total_bytes: int) -> None:
        remaining = total_bytes
        last_check = time.perf_counter()
        while remaining > 0:
            data = connection.recv(min(CHUNK_SIZE_BYTES, remaining))
            if not data:
                raise ConnectionError("Connection closed before transfer completed")
            remaining -= len(data)
            now = time.perf_counter()
            self.maybe_fail(now - last_check)
            last_check = now

    def _receive_exact_with_failures(self, connection: socket.socket, total_bytes: int) -> bytes:
        parts: list[bytes] = []
        remaining = total_bytes
        last_check = time.perf_counter()
        while remaining > 0:
            data = connection.recv(remaining)
            if not data:
                raise ConnectionError("Connection closed before header completed")
            parts.append(data)
            remaining -= len(data)
            now = time.perf_counter()
            self.maybe_fail(now - last_check)
            last_check = now
        return b"".join(parts)

    # ------------------------------------------------------------------
    # Result export
    # ------------------------------------------------------------------

    def write_result(
        self,
        result_file: Path,
        target_iterations: int,
        checkpoint_interval: int,
    ) -> None:
        runtime_s = time.perf_counter() - self.worker_start_time
        result = {
            "experiment": self.experiment,
            "variant": self.variant,
            "attempt": self.attempt,
            "host": self.name,
            "rank": self.rank,
            "world_size": self.world_size,
            "runtime_s": runtime_s,
            "target_iterations": target_iterations,
            "resume_checkpoint_iteration": self.resume_checkpoint_iteration,
            "last_completed_iteration": self.last_completed_iteration,
            "checkpoint_interval": checkpoint_interval,
            "configuration": {
                "gradient_size_mb": self.gradient_size_mb,
                "checkpoint_size_mb": self.checkpoint_size_mb,
                "checkpoint_wire_scale": self.checkpoint_wire_scale,
                "checkpoint_wire_mb": self.checkpoint_wire_bytes / 1024 / 1024,
                "network_bandwidth_mbps": self.network_bandwidth_mbps,
                "process_failure_percent_per_second": self.process_failure_rate * 100,
                "node_failure_percent_per_second": self.node_failure_rate * 100,
            },
            "counters": self.counters,
            "iterations": [
                self.iteration_metrics[index]
                for index in sorted(self.iteration_metrics)
            ],
        }
        result_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = result_file.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
        temporary.replace(result_file)
        self.log(None, "RESULT", "WRITE", f"path={result_file}, runtime_s={runtime_s:.3f}")
