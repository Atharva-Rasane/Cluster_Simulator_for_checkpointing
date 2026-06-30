#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import random
import select
import socket
import struct
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from store_protocol import receive_json, send_json


CHUNK_SIZE_BYTES = 1024 * 1024
DEFAULT_GRADIENT_BUCKET_CAP_MB = 25.0
DEFAULT_COLLECTIVE_TIMEOUT_S = 300.0

RING_PROTOCOL_MAGIC = b"DTRN"
RING_PROTOCOL_VERSION = 1
RING_PHASE_REDUCE_SCATTER = 1
RING_PHASE_ALL_GATHER = 2

# magic, version, phase, iteration, bucket index, step, sender rank, payload bytes
RING_STEP_HEADER = struct.Struct("!4sBBIIIIQ")

PROCESS_FAILURE_EXIT_CODE = 20
NODE_FAILURE_EXIT_CODE = 21


class SimulatedFailure(RuntimeError):
    def __init__(
        self,
        failure_type: str,
        message: str,
        *,
        iteration: int | None = None,
        stage: str | None = None,
    ) -> None:
        super().__init__(message)
        self.failure_type = failure_type
        self.iteration = iteration
        self.stage = stage
        self.exit_code = (
            PROCESS_FAILURE_EXIT_CODE
            if failure_type == "process"
            else NODE_FAILURE_EXIT_CODE
        )


class DistributedTraining:
    """
    Distributed-training, checkpoint, failure, and recovery procedures.

    This class deliberately performs real socket transfers so Mininet controls
    network delay, bandwidth, contention, and loss. Gradient contents are not
    material to the comparative simulator, so payloads contain zero bytes and
    received payloads are discarded.

    Gradient synchronization uses a persistent TCP ring with a reduce-scatter
    phase followed by an all-gather phase. Gradients are divided into buckets,
    and bucket communication can overlap the simulated backward pass.
    """

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
        gradient_bucket_cap_mb: float = DEFAULT_GRADIENT_BUCKET_CAP_MB,
        overlap_gradient_sync: bool = True,
        collective_timeout_s: float = DEFAULT_COLLECTIVE_TIMEOUT_S,
        ring_port: int | None = None,
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

        # This value is descriptive. Mininet is responsible for enforcing it.
        self.network_bandwidth_mbps = network_bandwidth_mbps

        self.forward_time = forward_time
        self.backward_time = backward_time
        self.update_time = update_time
        self.gradient_size_mb = gradient_size_mb
        self.gradient_bucket_cap_mb = gradient_bucket_cap_mb
        self.overlap_gradient_sync = overlap_gradient_sync

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

        self.collective_timeout_s = collective_timeout_s
        self.ring_port = (
            ring_port
            if ring_port is not None
            else self.master_port + self.rank + 1
        )

        self.worker_start_time = time.perf_counter()
        self.current_iteration = 0
        self.current_stage = "startup"
        self.last_completed_iteration = 0
        self.resume_checkpoint_iteration = 0

        self.iteration_metrics: dict[int, dict[str, Any]] = {}
        self.counters: dict[str, int | float] = {
            "gradient_bytes_sent": 0,
            "gradient_bytes_received": 0,
            "gradient_bucket_count": 0,
            "checkpoint_count": 0,
            "checkpoint_logical_bytes": 0,
            "checkpoint_wire_bytes_sent": 0,
            "recovery_requests": 0,
            "recovery_hits": 0,
            "recovery_misses": 0,
            "recovery_wire_bytes_received": 0,
            "recovery_total_s": 0.0,
        }

        self._state_lock = threading.Lock()
        self._metrics_lock = threading.Lock()
        self._random_lock = threading.Lock()
        self._log_lock = threading.Lock()
        self._stage_local = threading.local()

        self._collective_init_lock = threading.RLock()
        self._collective_initialized = False
        self._collective_closed = False
        self._ring_listener: socket.socket | None = None
        self._ring_send_socket: socket.socket | None = None
        self._ring_receive_socket: socket.socket | None = None
        self._ring_next_rank: int | None = None
        self._ring_previous_rank: int | None = None
        self._membership: dict[int, tuple[str, int]] = {}

        # A single communication worker preserves collective ordering while
        # allowing communication to overlap backward computation.
        self._collective_executor: ThreadPoolExecutor | None = None
        if self.world_size > 1:
            self._collective_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix=f"gradient-rank-{self.rank}",
            )

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
        if not 1 <= self.ring_port <= 65535:
            raise ValueError("ring_port must be between 1 and 65535")
        if self.rank == 0 and self.ring_port == self.master_port:
            raise ValueError("rank 0 ring_port must differ from master_port")
        if not 0 < self.checkpoint_wire_scale <= 1:
            raise ValueError("checkpoint_wire_scale must be in (0, 1]")

        positive = {
            "network_bandwidth_mbps": self.network_bandwidth_mbps,
            "gradient_size_mb": self.gradient_size_mb,
            "gradient_bucket_cap_mb": self.gradient_bucket_cap_mb,
            "checkpoint_size_mb": self.checkpoint_size_mb,
            "dram_capacity_mb": self.dram_capacity_mb,
            "gpu_to_dram_bandwidth_mb_s": self.gpu_to_dram_bandwidth_mb_s,
            "ssd_capacity_mb": self.ssd_capacity_mb,
            "ssd_bandwidth_mb_s": self.ssd_bandwidth_mb_s,
            "collective_timeout_s": self.collective_timeout_s,
        }
        for key, value in positive.items():
            if value <= 0:
                raise ValueError(f"{key} must be positive")

        for key, value in {
            "forward_time": self.forward_time,
            "backward_time": self.backward_time,
            "update_time": self.update_time,
        }.items():
            if value < 0:
                raise ValueError(f"{key} cannot be negative")

        for key, value in {
            "process_failure_rate": self.process_failure_rate,
            "node_failure_rate": self.node_failure_rate,
        }.items():
            if not 0 <= value < 1:
                raise ValueError(f"{key} must be in [0, 1)")

    def _thread_iteration_and_stage(self) -> tuple[int, str]:
        iteration = getattr(self._stage_local, "iteration", None)
        stage = getattr(self._stage_local, "stage", None)
        if iteration is not None and stage is not None:
            return int(iteration), str(stage)
        with self._state_lock:
            return self.current_iteration, self.current_stage

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
        with self._log_lock:
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
                f"ring_port={self.ring_port}, "
                f"store={self.object_store_ip}:{self.object_store_port}, "
                f"gradient_mb={self.gradient_size_mb:.2f}, "
                f"bucket_cap_mb={self.gradient_bucket_cap_mb:.2f}, "
                f"overlap_gradient_sync={self.overlap_gradient_sync}, "
                f"collective=ring-reduce-scatter-all-gather, "
                f"mininet_link_mbps={self.network_bandwidth_mbps:.2f}, "
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

    @property
    def gradient_bytes(self) -> int:
        return max(1, round(self.gradient_size_mb * 1024 * 1024))

    @property
    def gradient_bucket_cap_bytes(self) -> int:
        return max(1, round(self.gradient_bucket_cap_mb * 1024 * 1024))

    # ------------------------------------------------------------------
    # Failure injection
    # ------------------------------------------------------------------

    def set_stage(self, iteration: int, stage: str) -> None:
        self._stage_local.iteration = iteration
        self._stage_local.stage = stage
        with self._state_lock:
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

        # Treat process and node failures as independent events during the
        # elapsed interval. If both occur, report the node failure.
        node_only_or_both = node_probability
        process_only = (1.0 - node_probability) * process_probability

        with self._random_lock:
            draw = self.random.random()

        iteration, stage = self._thread_iteration_and_stage()
        if draw < node_only_or_both:
            raise SimulatedFailure(
                "node",
                f"simulated node failure at iter={iteration}, stage={stage}",
                iteration=iteration,
                stage=stage,
            )
        if draw < node_only_or_both + process_only:
            raise SimulatedFailure(
                "process",
                f"simulated process failure at iter={iteration}, stage={stage}",
                iteration=iteration,
                stage=stage,
            )

    def failure_aware_sleep(self, duration_s: float, check_interval_s: float = 0.1) -> None:
        remaining = duration_s
        while remaining > 0:
            step = min(check_interval_s, remaining)
            time.sleep(step)
            self.maybe_fail(step)
            remaining -= step

    def write_failure_event(self, failure: SimulatedFailure) -> None:
        failure_iteration = (
            failure.iteration
            if failure.iteration is not None
            else self.current_iteration
        )
        failure_stage = failure.stage or self.current_stage
        event = {
            "type": "worker_failure",
            "experiment": self.experiment,
            "variant": self.variant,
            "attempt": self.attempt,
            "host": self.name,
            "rank": self.rank,
            "failure_type": failure.failure_type,
            "exit_code": failure.exit_code,
            "iteration": failure_iteration,
            "stage": failure_stage,
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
            failure_iteration or None,
            "FAILURE",
            failure.failure_type.upper(),
            (
                f"stage={failure_stage}, last_completed={self.last_completed_iteration}, "
                f"event_file={self.failure_event_file}"
            ),
        )

    # ------------------------------------------------------------------
    # Recovery from external object store
    # ------------------------------------------------------------------

    def recover_from_object_store(self) -> int:
        self.set_stage(0, "recovery/get-checkpoint")
        with self._metrics_lock:
            self.counters["recovery_requests"] = int(self.counters["recovery_requests"]) + 1
        start = time.perf_counter()
        self.log(None, "RECOVERY", "START", "requesting latest object-store checkpoint")

        connection = self._connect(self.object_store_ip, self.object_store_port)
        with connection:
            send_json(
                connection,
                {
                    "op": "GET",
                    "experiment": self.experiment,
                    "variant": self.variant,
                    "attempt": self.attempt,
                    "rank": self.rank,
                },
            )
            response = receive_json(connection)

            if response.get("status") == "NOT_FOUND":
                duration = time.perf_counter() - start
                with self._metrics_lock:
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

            with self._metrics_lock:
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
        with self._metrics_lock:
            return self.iteration_metrics.setdefault(
                iteration,
                {
                    "iteration": iteration,
                    "forward_s": 0.0,
                    "backward_s": 0.0,
                    "gradient_sync_s": 0.0,
                    "gradient_wait_s": 0.0,
                    "gradient_overlap_s": 0.0,
                    "gradient_bucket_count": 0,
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
        record = self._record(iteration)
        with self._metrics_lock:
            record["_start"] = time.perf_counter()
        self.log(iteration, "ITERATION", "START", f"target={target_iterations}")

    def end_iteration(self, iteration: int, target_iterations: int) -> None:
        record = self._record(iteration)
        with self._metrics_lock:
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
        record = self._record(iteration)
        with self._metrics_lock:
            record["forward_s"] = duration
        self.log(iteration, "TRAIN/FORWARD", "END", f"duration_s={duration:.3f}")

    def backward_pass(self, iteration: int) -> None:
        """Run only the simulated backward compute, without gradient synchronization."""
        self.set_stage(iteration, "training/backward")
        start = time.perf_counter()
        self.log(iteration, "TRAIN/BACKWARD", "START", f"planned_s={self.backward_time:.3f}")
        self.failure_aware_sleep(self.backward_time)
        duration = time.perf_counter() - start
        record = self._record(iteration)
        with self._metrics_lock:
            record["backward_s"] = duration
        self.log(iteration, "TRAIN/BACKWARD", "END", f"duration_s={duration:.3f}")

    def _gradient_bucket_sizes(self) -> list[int]:
        total_bytes = self.gradient_bytes
        cap_bytes = self.gradient_bucket_cap_bytes
        bucket_sizes: list[int] = []
        remaining = total_bytes
        while remaining > 0:
            bucket_size = min(cap_bytes, remaining)
            bucket_sizes.append(bucket_size)
            remaining -= bucket_size
        return bucket_sizes

    def gradient_synchronization(self, iteration: int) -> None:
        """Synchronize the complete gradient as one collective operation."""
        self.set_stage(iteration, "training/gradient-sync")
        record = self._record(iteration)
        if self.world_size == 1:
            self.log(iteration, "GRADIENT_SYNC", "SKIP", "world_size=1")
            return

        result = self._synchronize_gradient_bucket(
            iteration=iteration,
            bucket_index=0,
            bucket_count=1,
            bucket_bytes=self.gradient_bytes,
        )
        self._apply_gradient_results(iteration, [result], backward_window=None)

    def _backward_with_bucketed_gradient_sync(self, iteration: int) -> None:
        if self.world_size == 1:
            self.backward_pass(iteration)
            self.log(iteration, "GRADIENT_SYNC", "SKIP", "world_size=1")
            return

        bucket_sizes = self._gradient_bucket_sizes()
        bucket_count = len(bucket_sizes)

        if not self.overlap_gradient_sync:
            self.backward_pass(iteration)
            results = [
                self._synchronize_gradient_bucket(
                    iteration=iteration,
                    bucket_index=bucket_index,
                    bucket_count=bucket_count,
                    bucket_bytes=bucket_bytes,
                )
                for bucket_index, bucket_bytes in enumerate(bucket_sizes)
            ]
            self._apply_gradient_results(iteration, results, backward_window=None)
            return

        if self._collective_executor is None:
            raise RuntimeError("Collective executor is not available")

        self.set_stage(iteration, "training/backward")
        backward_start = time.perf_counter()
        self.log(
            iteration,
            "TRAIN/BACKWARD",
            "START",
            (
                f"planned_s={self.backward_time:.3f}, buckets={bucket_count}, "
                "readiness=uniform"
            ),
        )

        futures: list[Future[dict[str, int | float]]] = []
        completed_compute_s = 0.0

        try:
            for bucket_index, bucket_bytes in enumerate(bucket_sizes):
                target_compute_s = self.backward_time * (bucket_index + 1) / bucket_count
                compute_slice_s = max(0.0, target_compute_s - completed_compute_s)
                self.failure_aware_sleep(compute_slice_s)
                completed_compute_s = target_compute_s

                self.log(
                    iteration,
                    "GRADIENT/BUCKET",
                    "READY",
                    (
                        f"bucket={bucket_index + 1}/{bucket_count}, "
                        f"mb={bucket_bytes / 1024 / 1024:.2f}"
                    ),
                )
                futures.append(
                    self._collective_executor.submit(
                        self._synchronize_gradient_bucket,
                        iteration,
                        bucket_index,
                        bucket_count,
                        bucket_bytes,
                    )
                )

            backward_end = time.perf_counter()
            backward_duration = backward_end - backward_start
            record = self._record(iteration)
            with self._metrics_lock:
                record["backward_s"] = backward_duration
            self.log(
                iteration,
                "TRAIN/BACKWARD",
                "END",
                f"duration_s={backward_duration:.3f}",
            )

            wait_start = time.perf_counter()
            results = [future.result() for future in futures]
            wait_end = time.perf_counter()
            self._apply_gradient_results(
                iteration,
                results,
                backward_window=(backward_start, backward_end, wait_start, wait_end),
            )
        except BaseException:
            for future in futures:
                future.cancel()
            raise

    def _synchronize_gradient_bucket(
        self,
        iteration: int,
        bucket_index: int,
        bucket_count: int,
        bucket_bytes: int,
    ) -> dict[str, int | float]:
        self.set_stage(
            iteration,
            f"training/gradient-sync/bucket-{bucket_index + 1}",
        )
        self._ensure_collective_initialized()

        start = time.perf_counter()
        self.log(
            iteration,
            "GRADIENT_SYNC",
            "START",
            (
                f"bucket={bucket_index + 1}/{bucket_count}, "
                f"payload_mb={bucket_bytes / 1024 / 1024:.2f}, "
                "algorithm=ring"
            ),
        )

        try:
            sent, received = self._ring_all_reduce(
                iteration=iteration,
                bucket_index=bucket_index,
                bucket_bytes=bucket_bytes,
            )
        except BaseException:
            self._invalidate_collective_connections()
            raise

        end = time.perf_counter()
        duration = end - start
        self.log(
            iteration,
            "GRADIENT_SYNC",
            "END",
            (
                f"bucket={bucket_index + 1}/{bucket_count}, "
                f"duration_s={duration:.3f}, "
                f"sent_mb={sent / 1024 / 1024:.2f}, "
                f"received_mb={received / 1024 / 1024:.2f}"
            ),
        )
        return {
            "start": start,
            "end": end,
            "duration": duration,
            "sent": sent,
            "received": received,
        }

    def _apply_gradient_results(
        self,
        iteration: int,
        results: list[dict[str, int | float]],
        backward_window: tuple[float, float, float, float] | None,
    ) -> None:
        if not results:
            return

        first_start = min(float(result["start"]) for result in results)
        last_end = max(float(result["end"]) for result in results)
        communication_span = last_end - first_start
        sent = sum(int(result["sent"]) for result in results)
        received = sum(int(result["received"]) for result in results)

        wait_s = 0.0
        overlap_s = 0.0
        if backward_window is not None:
            backward_start, backward_end, wait_start, wait_end = backward_window
            wait_s = max(0.0, wait_end - wait_start)
            overlap_s = max(
                0.0,
                min(last_end, backward_end) - max(first_start, backward_start),
            )

        record = self._record(iteration)
        with self._metrics_lock:
            record["gradient_sync_s"] = communication_span
            record["gradient_wait_s"] = wait_s
            record["gradient_overlap_s"] = overlap_s
            record["gradient_bucket_count"] = len(results)
            record["gradient_bytes_sent"] = sent
            record["gradient_bytes_received"] = received
            self.counters["gradient_bytes_sent"] = int(self.counters["gradient_bytes_sent"]) + sent
            self.counters["gradient_bytes_received"] = int(self.counters["gradient_bytes_received"]) + received
            self.counters["gradient_bucket_count"] = int(self.counters["gradient_bucket_count"]) + len(results)

        self.log(
            iteration,
            "GRADIENT_SYNC",
            "SUMMARY",
            (
                f"buckets={len(results)}, span_s={communication_span:.3f}, "
                f"overlap_s={overlap_s:.3f}, wait_s={wait_s:.3f}, "
                f"sent_mb={sent / 1024 / 1024:.2f}, "
                f"received_mb={received / 1024 / 1024:.2f}"
            ),
        )

    def training_stage(self, iteration: int) -> None:
        start = time.perf_counter()
        self.log(iteration, "TRAINING", "START")
        self.forward_pass(iteration)
        self._backward_with_bucketed_gradient_sync(iteration)
        duration = time.perf_counter() - start
        record = self._record(iteration)
        with self._metrics_lock:
            record["training_s"] = duration
        self.log(iteration, "TRAINING", "END", f"duration_s={duration:.3f}")

    # ------------------------------------------------------------------
    # Persistent ring collective
    # ------------------------------------------------------------------

    def initialize_collective(self) -> None:
        """Optionally initialize persistent collective connections before training."""
        if self.world_size > 1:
            self._ensure_collective_initialized()

    def _ensure_collective_initialized(self) -> None:
        if self.world_size == 1 or self._collective_initialized:
            return

        with self._collective_init_lock:
            if self._collective_initialized:
                return
            if self._collective_closed:
                raise RuntimeError("Collective has already been closed")

            self.set_stage(self.current_iteration, "collective/rendezvous")
            self.log(
                self.current_iteration or None,
                "COLLECTIVE",
                "INIT",
                (
                    f"algorithm=ring, local={self.local_ip}:{self.ring_port}, "
                    f"master={self.master_ip}:{self.master_port}"
                ),
            )

            try:
                self._ring_listener = self._create_server(
                    host=self.local_ip,
                    port=self.ring_port,
                    backlog=max(1, self.world_size),
                )

                if self.rank == 0:
                    membership = self._rank_zero_rendezvous()
                else:
                    membership = self._nonzero_rendezvous()

                self._membership = membership
                self._connect_ring_neighbors()
                self._collective_initialized = True
                self.log(
                    self.current_iteration or None,
                    "COLLECTIVE",
                    "READY",
                    (
                        f"previous_rank={self._ring_previous_rank}, "
                        f"next_rank={self._ring_next_rank}"
                    ),
                )
            except BaseException:
                self._invalidate_collective_connections()
                raise

    def _rank_zero_rendezvous(self) -> dict[int, tuple[str, int]]:
        membership: dict[int, tuple[str, int]] = {
            0: (self.local_ip, self.ring_port)
        }
        registration_connections: dict[int, socket.socket] = {}

        rendezvous_server = self._create_server(
            host=self.local_ip,
            port=self.master_port,
            backlog=max(1, self.world_size - 1),
        )
        try:
            while len(membership) < self.world_size:
                connection, address = self._accept_with_failures(rendezvous_server)
                try:
                    registration = receive_json(connection)
                    peer_rank = int(registration.get("rank", -1))
                    peer_world_size = int(registration.get("world_size", -1))
                    peer_ip = str(registration.get("local_ip", address[0]))
                    peer_ring_port = int(registration.get("ring_port", -1))

                    if registration.get("op") != "REGISTER_RING":
                        raise RuntimeError(f"Invalid rendezvous request: {registration}")
                    if registration.get("experiment") != self.experiment:
                        raise RuntimeError("Rendezvous experiment mismatch")
                    if registration.get("variant") != self.variant:
                        raise RuntimeError("Rendezvous variant mismatch")
                    if int(registration.get("attempt", -1)) != self.attempt:
                        raise RuntimeError("Rendezvous attempt mismatch")
                    if peer_world_size != self.world_size:
                        raise RuntimeError("Rendezvous world_size mismatch")
                    if not 1 <= peer_rank < self.world_size:
                        raise RuntimeError(f"Invalid peer rank: {peer_rank}")
                    if peer_rank in membership:
                        raise RuntimeError(f"Duplicate peer rank: {peer_rank}")
                    if not 1 <= peer_ring_port <= 65535:
                        raise RuntimeError(f"Invalid peer ring port: {peer_ring_port}")

                    connection.settimeout(self.collective_timeout_s)
                    membership[peer_rank] = (peer_ip, peer_ring_port)
                    registration_connections[peer_rank] = connection
                    self.log(
                        self.current_iteration or None,
                        "COLLECTIVE/RDZV",
                        "REGISTER",
                        (
                            f"peer_rank={peer_rank}, peer={peer_ip}:{peer_ring_port}, "
                            f"registered={len(membership)}/{self.world_size}"
                        ),
                    )
                except BaseException:
                    connection.close()
                    raise

            members_payload = [
                {
                    "rank": rank,
                    "ip": membership[rank][0],
                    "ring_port": membership[rank][1],
                }
                for rank in range(self.world_size)
            ]
            response = {
                "status": "OK",
                "op": "RING_MEMBERSHIP",
                "experiment": self.experiment,
                "variant": self.variant,
                "attempt": self.attempt,
                "world_size": self.world_size,
                "members": members_payload,
            }
            for peer_rank in sorted(registration_connections):
                connection = registration_connections[peer_rank]
                send_json(connection, response)
                connection.close()
            registration_connections.clear()
            return membership
        finally:
            for connection in registration_connections.values():
                connection.close()
            rendezvous_server.close()

    def _nonzero_rendezvous(self) -> dict[int, tuple[str, int]]:
        connection = self._connect(
            self.master_ip,
            self.master_port,
            retry=True,
            timeout_s=self.collective_timeout_s,
        )
        with connection:
            send_json(
                connection,
                {
                    "op": "REGISTER_RING",
                    "experiment": self.experiment,
                    "variant": self.variant,
                    "attempt": self.attempt,
                    "rank": self.rank,
                    "world_size": self.world_size,
                    "local_ip": self.local_ip,
                    "ring_port": self.ring_port,
                },
            )
            response = receive_json(connection)

        if response.get("status") != "OK" or response.get("op") != "RING_MEMBERSHIP":
            raise RuntimeError(f"Invalid rendezvous response: {response}")
        if response.get("experiment") != self.experiment:
            raise RuntimeError("Rendezvous response experiment mismatch")
        if response.get("variant") != self.variant:
            raise RuntimeError("Rendezvous response variant mismatch")
        if int(response.get("attempt", -1)) != self.attempt:
            raise RuntimeError("Rendezvous response attempt mismatch")
        if int(response.get("world_size", -1)) != self.world_size:
            raise RuntimeError("Rendezvous response world_size mismatch")

        membership: dict[int, tuple[str, int]] = {}
        for member in response.get("members", []):
            member_rank = int(member["rank"])
            member_ip = str(member["ip"])
            member_port = int(member["ring_port"])
            if member_rank in membership:
                raise RuntimeError(f"Duplicate membership rank: {member_rank}")
            membership[member_rank] = (member_ip, member_port)

        if set(membership) != set(range(self.world_size)):
            raise RuntimeError("Rendezvous membership is incomplete")
        return membership

    def _connect_ring_neighbors(self) -> None:
        if self._ring_listener is None:
            raise RuntimeError("Ring listener is not initialized")

        previous_rank = (self.rank - 1) % self.world_size
        next_rank = (self.rank + 1) % self.world_size
        next_ip, next_port = self._membership[next_rank]

        outgoing = self._connect(
            next_ip,
            next_port,
            retry=True,
            timeout_s=self.collective_timeout_s,
        )
        try:
            send_json(
                outgoing,
                {
                    "op": "RING_CONNECT",
                    "experiment": self.experiment,
                    "variant": self.variant,
                    "attempt": self.attempt,
                    "world_size": self.world_size,
                    "rank": self.rank,
                },
            )

            incoming, _ = self._accept_with_failures(self._ring_listener)
            try:
                handshake = receive_json(incoming)
                if handshake.get("op") != "RING_CONNECT":
                    raise RuntimeError(f"Invalid ring handshake: {handshake}")
                if handshake.get("experiment") != self.experiment:
                    raise RuntimeError("Ring handshake experiment mismatch")
                if handshake.get("variant") != self.variant:
                    raise RuntimeError("Ring handshake variant mismatch")
                if int(handshake.get("attempt", -1)) != self.attempt:
                    raise RuntimeError("Ring handshake attempt mismatch")
                if int(handshake.get("world_size", -1)) != self.world_size:
                    raise RuntimeError("Ring handshake world_size mismatch")
                if int(handshake.get("rank", -1)) != previous_rank:
                    raise RuntimeError(
                        f"Expected previous rank {previous_rank}, "
                        f"received {handshake.get('rank')}"
                    )
            except BaseException:
                incoming.close()
                raise
        except BaseException:
            outgoing.close()
            raise
        finally:
            self._ring_listener.close()
            self._ring_listener = None

        outgoing.settimeout(self.collective_timeout_s)
        incoming.settimeout(self.collective_timeout_s)
        self._ring_send_socket = outgoing
        self._ring_receive_socket = incoming
        self._ring_next_rank = next_rank
        self._ring_previous_rank = previous_rank

    def _ring_all_reduce(
        self,
        iteration: int,
        bucket_index: int,
        bucket_bytes: int,
    ) -> tuple[int, int]:
        if self.world_size <= 1:
            return 0, 0
        if self._ring_send_socket is None or self._ring_receive_socket is None:
            raise RuntimeError("Ring sockets are not initialized")

        chunks = self._split_bytes(bucket_bytes, self.world_size)
        sent_total = 0
        received_total = 0

        for step in range(self.world_size - 1):
            send_chunk_index = (self.rank - step) % self.world_size
            receive_chunk_index = (self.rank - step - 1) % self.world_size
            sent, received = self._ring_exchange_step(
                iteration=iteration,
                bucket_index=bucket_index,
                phase=RING_PHASE_REDUCE_SCATTER,
                step=step,
                send_bytes=chunks[send_chunk_index],
                expected_receive_bytes=chunks[receive_chunk_index],
            )
            sent_total += sent
            received_total += received

        for step in range(self.world_size - 1):
            send_chunk_index = (self.rank - step + 1) % self.world_size
            receive_chunk_index = (self.rank - step) % self.world_size
            sent, received = self._ring_exchange_step(
                iteration=iteration,
                bucket_index=bucket_index,
                phase=RING_PHASE_ALL_GATHER,
                step=step,
                send_bytes=chunks[send_chunk_index],
                expected_receive_bytes=chunks[receive_chunk_index],
            )
            sent_total += sent
            received_total += received

        return sent_total, received_total

    def _ring_exchange_step(
        self,
        iteration: int,
        bucket_index: int,
        phase: int,
        step: int,
        send_bytes: int,
        expected_receive_bytes: int,
    ) -> tuple[int, int]:
        if self._ring_send_socket is None or self._ring_receive_socket is None:
            raise RuntimeError("Ring sockets are not initialized")
        if self._ring_previous_rank is None:
            raise RuntimeError("Previous ring rank is not initialized")

        header = RING_STEP_HEADER.pack(
            RING_PROTOCOL_MAGIC,
            RING_PROTOCOL_VERSION,
            phase,
            iteration,
            bucket_index,
            step,
            self.rank,
            send_bytes,
        )
        self._send_exact_with_failures(self._ring_send_socket, header)

        received_header = self._receive_exact_with_failures(
            self._ring_receive_socket,
            RING_STEP_HEADER.size,
        )
        (
            magic,
            version,
            received_phase,
            received_iteration,
            received_bucket_index,
            received_step,
            sender_rank,
            received_payload_bytes,
        ) = RING_STEP_HEADER.unpack(received_header)

        if magic != RING_PROTOCOL_MAGIC:
            raise RuntimeError("Invalid ring protocol magic")
        if version != RING_PROTOCOL_VERSION:
            raise RuntimeError("Unsupported ring protocol version")
        if received_phase != phase:
            raise RuntimeError("Ring phase mismatch")
        if received_iteration != iteration:
            raise RuntimeError("Ring iteration mismatch")
        if received_bucket_index != bucket_index:
            raise RuntimeError("Ring bucket mismatch")
        if received_step != step:
            raise RuntimeError("Ring step mismatch")
        if sender_rank != self._ring_previous_rank:
            raise RuntimeError("Ring sender-rank mismatch")
        if received_payload_bytes != expected_receive_bytes:
            raise RuntimeError(
                "Ring payload-size mismatch: "
                f"expected={expected_receive_bytes}, received={received_payload_bytes}"
            )

        return self._exchange_zero_and_discard_with_failures(
            send_connection=self._ring_send_socket,
            receive_connection=self._ring_receive_socket,
            send_bytes=send_bytes,
            receive_bytes=expected_receive_bytes,
        )

    @staticmethod
    def _split_bytes(total_bytes: int, part_count: int) -> list[int]:
        base, remainder = divmod(total_bytes, part_count)
        return [
            base + (1 if index < remainder else 0)
            for index in range(part_count)
        ]

    def _invalidate_collective_connections(self) -> None:
        with self._collective_init_lock:
            for connection in (
                self._ring_send_socket,
                self._ring_receive_socket,
                self._ring_listener,
            ):
                if connection is not None:
                    try:
                        connection.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                    try:
                        connection.close()
                    except OSError:
                        pass
            self._ring_send_socket = None
            self._ring_receive_socket = None
            self._ring_listener = None
            self._ring_next_rank = None
            self._ring_previous_rank = None
            self._membership = {}
            self._collective_initialized = False

    # ------------------------------------------------------------------
    # Update stage
    # ------------------------------------------------------------------

    def update_stage(self, iteration: int) -> None:
        self.set_stage(iteration, "update")
        start = time.perf_counter()
        self.log(iteration, "UPDATE", "START", f"planned_s={self.update_time:.3f}")
        self.failure_aware_sleep(self.update_time)
        duration = time.perf_counter() - start
        record = self._record(iteration)
        with self._metrics_lock:
            record["update_s"] = duration
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
        with self._metrics_lock:
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
                    "variant": self.variant,
                    "attempt": self.attempt,
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
        with self._metrics_lock:
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
        start = time.perf_counter()
        self.failure_aware_sleep(duration)
        actual_duration = time.perf_counter() - start
        self.log(iteration, "CHECKPOINT/DRAM", "END", f"duration_s={actual_duration:.3f}")

    def checkpoint_to_ssd(self, iteration: int) -> None:
        if self.checkpoint_size_mb > self.ssd_capacity_mb:
            raise RuntimeError("Checkpoint exceeds SSD capacity")
        self.checkpoint_to_dram(iteration)
        self.set_stage(iteration, "checkpoint/local-ssd")
        duration = self.checkpoint_size_mb / self.ssd_bandwidth_mb_s
        self.log(iteration, "CHECKPOINT/SSD", "START", f"expected_s={duration:.3f}")
        start = time.perf_counter()
        self.failure_aware_sleep(duration)
        actual_duration = time.perf_counter() - start
        self.log(iteration, "CHECKPOINT/SSD", "END", f"duration_s={actual_duration:.3f}")

    # ------------------------------------------------------------------
    # Socket helpers
    # ------------------------------------------------------------------

    def _create_server(self, host: str, port: int, backlog: int) -> socket.socket:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            server.bind((host, port))
            server.listen(backlog)
            server.settimeout(min(1.0, self.collective_timeout_s))
            return server
        except BaseException:
            server.close()
            raise

    def _accept_with_failures(
        self,
        server: socket.socket,
    ) -> tuple[socket.socket, tuple[str, int]]:
        deadline = time.monotonic() + self.collective_timeout_s
        last_check = time.perf_counter()
        while True:
            try:
                connection, address = server.accept()
                connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                connection.settimeout(self.collective_timeout_s)
                return connection, address
            except socket.timeout:
                now = time.perf_counter()
                self.maybe_fail(now - last_check)
                last_check = now
                if time.monotonic() >= deadline:
                    raise TimeoutError("Timed out waiting for collective connection")

    def _connect(
        self,
        host: str,
        port: int,
        retry: bool = False,
        timeout_s: float | None = None,
    ) -> socket.socket:
        effective_timeout = timeout_s or self.collective_timeout_s
        deadline = time.monotonic() + effective_timeout
        last_check = time.perf_counter()
        while True:
            connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            connection.settimeout(min(5.0, effective_timeout))
            try:
                connection.connect((host, port))
                connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                connection.settimeout(effective_timeout)
                return connection
            except OSError as error:
                connection.close()
                now = time.perf_counter()
                self.maybe_fail(now - last_check)
                last_check = now
                if not retry or time.monotonic() >= deadline:
                    raise ConnectionError(f"Could not connect to {host}:{port}") from error
                self.failure_aware_sleep(0.05, check_interval_s=0.05)

    def _send_exact_with_failures(
        self,
        connection: socket.socket,
        payload: bytes,
    ) -> None:
        view = memoryview(payload)
        sent = 0
        last_check = time.perf_counter()
        while sent < len(payload):
            try:
                amount = connection.send(view[sent:])
            except socket.timeout as error:
                raise TimeoutError("Timed out sending socket payload") from error
            if amount <= 0:
                raise ConnectionError("Connection closed while sending payload")
            sent += amount
            now = time.perf_counter()
            self.maybe_fail(now - last_check)
            last_check = now

    def _send_zero_with_failures(self, connection: socket.socket, total_bytes: int) -> None:
        chunk = b"\0" * CHUNK_SIZE_BYTES
        view = memoryview(chunk)
        remaining = total_bytes
        last_check = time.perf_counter()
        while remaining > 0:
            amount = min(CHUNK_SIZE_BYTES, remaining)
            try:
                sent = connection.send(view[:amount])
            except socket.timeout as error:
                raise TimeoutError("Timed out sending transfer payload") from error
            if sent <= 0:
                raise ConnectionError("Connection closed while sending transfer payload")
            remaining -= sent
            now = time.perf_counter()
            self.maybe_fail(now - last_check)
            last_check = now

    def _receive_discard_with_failures(self, connection: socket.socket, total_bytes: int) -> None:
        remaining = total_bytes
        last_check = time.perf_counter()
        while remaining > 0:
            try:
                data = connection.recv(min(CHUNK_SIZE_BYTES, remaining))
            except socket.timeout as error:
                raise TimeoutError("Timed out receiving transfer payload") from error
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
            try:
                data = connection.recv(remaining)
            except socket.timeout as error:
                raise TimeoutError("Timed out receiving socket payload") from error
            if not data:
                raise ConnectionError("Connection closed before payload completed")
            parts.append(data)
            remaining -= len(data)
            now = time.perf_counter()
            self.maybe_fail(now - last_check)
            last_check = now
        return b"".join(parts)

    def _exchange_zero_and_discard_with_failures(
        self,
        send_connection: socket.socket,
        receive_connection: socket.socket,
        send_bytes: int,
        receive_bytes: int,
    ) -> tuple[int, int]:
        """Send and receive concurrently to avoid ring-wide send/send deadlocks."""
        zero_chunk = b"\0" * CHUNK_SIZE_BYTES
        zero_view = memoryview(zero_chunk)
        send_remaining = send_bytes
        receive_remaining = receive_bytes
        sent_total = 0
        received_total = 0

        deadline = time.monotonic() + self.collective_timeout_s
        last_check = time.perf_counter()

        while send_remaining > 0 or receive_remaining > 0:
            remaining_timeout = deadline - time.monotonic()
            if remaining_timeout <= 0:
                raise TimeoutError("Timed out during ring payload exchange")

            read_list = [receive_connection] if receive_remaining > 0 else []
            write_list = [send_connection] if send_remaining > 0 else []
            try:
                readable, writable, exceptional = select.select(
                    read_list,
                    write_list,
                    read_list + write_list,
                    min(0.1, remaining_timeout),
                )
            except OSError as error:
                raise ConnectionError("Ring select failed") from error

            now = time.perf_counter()
            self.maybe_fail(now - last_check)
            last_check = now

            if exceptional:
                raise ConnectionError("Exceptional socket condition during ring exchange")

            if writable and send_remaining > 0:
                amount = min(CHUNK_SIZE_BYTES, send_remaining)
                try:
                    sent = send_connection.send(zero_view[:amount])
                except (BlockingIOError, InterruptedError):
                    sent = None
                except socket.timeout as error:
                    raise TimeoutError("Timed out sending ring payload") from error
                if sent is not None:
                    if sent <= 0:
                        raise ConnectionError("Ring send connection closed")
                    send_remaining -= sent
                    sent_total += sent

            if readable and receive_remaining > 0:
                try:
                    data = receive_connection.recv(
                        min(CHUNK_SIZE_BYTES, receive_remaining)
                    )
                except (BlockingIOError, InterruptedError):
                    data = None
                except socket.timeout as error:
                    raise TimeoutError("Timed out receiving ring payload") from error
                if data is not None:
                    if not data:
                        raise ConnectionError("Ring receive connection closed")
                    receive_remaining -= len(data)
                    received_total += len(data)

        return sent_total, received_total

    # ------------------------------------------------------------------
    # Result export and cleanup
    # ------------------------------------------------------------------

    def write_result(
        self,
        result_file: Path,
        target_iterations: int,
        checkpoint_interval: int,
    ) -> None:
        runtime_s = time.perf_counter() - self.worker_start_time
        with self._metrics_lock:
            iterations = [
                dict(self.iteration_metrics[index])
                for index in sorted(self.iteration_metrics)
            ]
            counters = dict(self.counters)

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
                "gradient_bucket_cap_mb": self.gradient_bucket_cap_mb,
                "overlap_gradient_sync": self.overlap_gradient_sync,
                "collective_algorithm": "ring-reduce-scatter-all-gather",
                "ring_port": self.ring_port,
                "checkpoint_size_mb": self.checkpoint_size_mb,
                "checkpoint_wire_scale": self.checkpoint_wire_scale,
                "checkpoint_wire_mb": self.checkpoint_wire_bytes / 1024 / 1024,
                "network_bandwidth_mbps": self.network_bandwidth_mbps,
                "network_bandwidth_enforced_by": "mininet",
                "process_failure_percent_per_second": self.process_failure_rate * 100,
                "node_failure_percent_per_second": self.node_failure_rate * 100,
            },
            "counters": counters,
            "iterations": iterations,
        }
        result_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = result_file.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
        temporary.replace(result_file)
        self.log(None, "RESULT", "WRITE", f"path={result_file}, runtime_s={runtime_s:.3f}")

    def close(self) -> None:
        if self._collective_closed:
            return
        self._collective_closed = True

        # Closing sockets first unblocks a communication worker that may be
        # waiting inside select(), send(), or recv().
        self._invalidate_collective_connections()

        if self._collective_executor is not None:
            self._collective_executor.shutdown(wait=True, cancel_futures=True)
            self._collective_executor = None

    def __enter__(self) -> DistributedTraining:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
