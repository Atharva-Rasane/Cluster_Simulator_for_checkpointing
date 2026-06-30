#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import shutil
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mininet.link import TCLink
from mininet.log import setLogLevel
from mininet.net import Mininet
from mininet.node import OVSBridge

from distributed_training import NODE_FAILURE_EXIT_CODE, PROCESS_FAILURE_EXIT_CODE


# =====================================================================
# USER CONFIGURATION
# =====================================================================

# Every worker is reachable from every other worker through the shared switch.
NUMBER_OF_NODES = 4
TARGET_ITERATIONS = 5

WORKER_VARIANTS_TO_RUN = [
    # "nocheckpointing",
    "checkpointing",
]

WORKER_SCRIPTS = {
    # "nocheckpointing": "worker-nocheckpointing.py",
    "checkpointing": "worker-checkpointing.py",
}

# ---------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------

NETWORK_BANDWIDTH_MBPS = 10_000.0
STORE_NETWORK_BANDWIDTH_MBPS = 10_000.0
LINK_DELAY_MS = 1

MASTER_IP = "10.0.0.1"
MASTER_PORT = 5000
OBJECT_STORE_IP = "10.0.0.250"
OBJECT_STORE_PORT = 7000

# ---------------------------------------------------------------------
# Training profile
# ---------------------------------------------------------------------

FORWARD_TIME_SECONDS = 3.0
BACKWARD_TIME_SECONDS = 5.0
UPDATE_TIME_SECONDS = 0.8

# These bytes are genuinely sent through the Mininet network on every
# gradient synchronization.
GRADIENT_SIZE_MB = 14*1024

# ---------------------------------------------------------------------
# Checkpoint profile
# ---------------------------------------------------------------------

CHECKPOINT_INTERVAL = 1

# Logical checkpoint size for the timing model: 16.2 GB.
CHECKPOINT_SIZE_MB = 96 * 1024

# The object-store host receives actual bytes. Sending a full 16.2 GB on every
# checkpoint is inconvenient for a laptop simulation, so the wire payload is
# scaled while the timing/resource model continues to use the logical size.
# Set this to 1.0 to send the complete checkpoint size.
CHECKPOINT_WIRE_SCALE = 0.001

DRAM_CAPACITY_MB = 512 * 1024
GPU_TO_DRAM_BANDWIDTH_MB_S = 12_000.0

# Kept for future local-SSD worker variants. The current checkpoint worker does
# not use SSD; it sends GPU -> DRAM -> external object store.
SSD_CAPACITY_MB = 4 * 1024 * 1024
SSD_BANDWIDTH_MB_S = 3_500.0

OBJECT_STORE_WRITE_BANDWIDTH_MB_S = 5_000.0
OBJECT_STORE_READ_BANDWIDTH_MB_S = 6_000.0

# ---------------------------------------------------------------------
# Failure injection and recovery
# ---------------------------------------------------------------------

# Hazard percentages checked throughout compute, communication, checkpoint,
# and recovery work. Values are percentages per second, not per iteration.
PROCESS_FAILURE_PERCENT_PER_SECOND = 0.09
NODE_FAILURE_PERCENT_PER_SECOND = 0.01

PROCESS_RESTART_TIME_SECONDS = 2.0
NODE_RESTART_TIME_SECONDS = 15.0
MAX_RECOVERY_ATTEMPTS = 25
RANDOM_SEED = 20260629

RESULTS_DIRECTORY_NAME = "results"


# =====================================================================
# CONFIGURATION AND DISPLAY HELPERS
# =====================================================================


def validate_configuration() -> None:
    if not 1 <= NUMBER_OF_NODES <= 249:
        raise ValueError("NUMBER_OF_NODES must be between 1 and 249")
    if TARGET_ITERATIONS < 1:
        raise ValueError("TARGET_ITERATIONS must be at least 1")
    if CHECKPOINT_INTERVAL < 1:
        raise ValueError("CHECKPOINT_INTERVAL must be at least 1")
    if not 0 < CHECKPOINT_WIRE_SCALE <= 1:
        raise ValueError("CHECKPOINT_WIRE_SCALE must be in (0, 1]")
    if CHECKPOINT_SIZE_MB > DRAM_CAPACITY_MB:
        raise ValueError("Checkpoint does not fit in DRAM staging capacity")
    unknown = set(WORKER_VARIANTS_TO_RUN) - set(WORKER_SCRIPTS)
    if unknown:
        raise ValueError(f"Unknown worker variants: {sorted(unknown)}")
    if MAX_RECOVERY_ATTEMPTS < 1:
        raise ValueError("MAX_RECOVERY_ATTEMPTS must be at least 1")


def create_host_configurations() -> list[dict[str, Any]]:
    return [
        {
            "name": f"h{rank + 1}",
            "rank": rank,
            "ip": f"10.0.0.{rank + 1}",
        }
        for rank in range(NUMBER_OF_NODES)
    ]


def print_configuration_summary() -> None:
    checkpoint_wire_mb = CHECKPOINT_SIZE_MB * CHECKPOINT_WIRE_SCALE
    gpu_to_dram_s = CHECKPOINT_SIZE_MB / GPU_TO_DRAM_BANDWIDTH_MB_S
    object_write_s = CHECKPOINT_SIZE_MB / OBJECT_STORE_WRITE_BANDWIDTH_MB_S
    object_read_s = CHECKPOINT_SIZE_MB / OBJECT_STORE_READ_BANDWIDTH_MB_S

    print("\n" + "=" * 96)
    print("SIMULATION CONFIGURATION")
    print("=" * 96)
    print(f"Worker variants:                    {', '.join(WORKER_VARIANTS_TO_RUN)}")
    print(f"Fully reachable worker nodes:       {NUMBER_OF_NODES}")
    print(f"Target completed iterations:        {TARGET_ITERATIONS}")
    print(f"Worker link bandwidth:              {NETWORK_BANDWIDTH_MBPS:.0f} Mbps")
    print(f"Object-store link bandwidth:        {STORE_NETWORK_BANDWIDTH_MBPS:.0f} Mbps")
    print(f"Link delay:                         {LINK_DELAY_MS} ms")
    print(f"Actual gradient payload/rank:       {GRADIENT_SIZE_MB:.2f} MB/iteration")
    print(f"Checkpoint interval:                every {CHECKPOINT_INTERVAL} iteration(s)")
    print(f"Logical checkpoint size:            {CHECKPOINT_SIZE_MB / 1024:.2f} GB")
    print(f"Actual checkpoint wire payload:     {checkpoint_wire_mb:.2f} MB")
    print(f"GPU->DRAM checkpoint time:          {gpu_to_dram_s:.3f}s")
    print(f"Object-store persist time:          {object_write_s:.3f}s")
    print(f"Object-store recovery read time:    {object_read_s:.3f}s")
    print(f"Process failure hazard:             {PROCESS_FAILURE_PERCENT_PER_SECOND:.4f}%/s")
    print(f"Node failure hazard:                {NODE_FAILURE_PERCENT_PER_SECOND:.4f}%/s")
    print(f"Process restart delay:              {PROCESS_RESTART_TIME_SECONDS:.3f}s")
    print(f"Node restart delay:                 {NODE_RESTART_TIME_SECONDS:.3f}s")
    print("=" * 96 + "\n", flush=True)


def print_table(headers: list[str], rows: list[list[Any]]) -> None:
    string_rows = [[str(value) for value in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in string_rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    print(separator)
    print("| " + " | ".join(header.ljust(widths[i]) for i, header in enumerate(headers)) + " |")
    print(separator)
    for row in string_rows:
        print("| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(row)) + " |")
    print(separator)


# =====================================================================
# COMMAND CONSTRUCTION
# =====================================================================


def create_worker_command(
    worker_script: Path,
    experiment: str,
    variant: str,
    attempt: int,
    host_configuration: dict[str, Any],
    result_file: Path,
    failure_event_file: Path,
) -> list[str]:
    return [
        "python3",
        "-u",
        str(worker_script),
        "--experiment",
        experiment,
        "--variant",
        variant,
        "--attempt",
        str(attempt),
        "--result-file",
        str(result_file),
        "--failure-event-file",
        str(failure_event_file),
        "--name",
        str(host_configuration["name"]),
        "--rank",
        str(host_configuration["rank"]),
        "--world-size",
        str(NUMBER_OF_NODES),
        "--local-ip",
        str(host_configuration["ip"]),
        "--master-ip",
        MASTER_IP,
        "--master-port",
        str(MASTER_PORT),
        "--object-store-ip",
        OBJECT_STORE_IP,
        "--object-store-port",
        str(OBJECT_STORE_PORT),
        "--network-bandwidth-mbps",
        str(NETWORK_BANDWIDTH_MBPS),
        "--target-iterations",
        str(TARGET_ITERATIONS),
        "--forward-time",
        str(FORWARD_TIME_SECONDS),
        "--backward-time",
        str(BACKWARD_TIME_SECONDS),
        "--update-time",
        str(UPDATE_TIME_SECONDS),
        "--gradient-size-mb",
        str(GRADIENT_SIZE_MB),
        "--checkpoint-interval",
        str(CHECKPOINT_INTERVAL),
        "--checkpoint-size-mb",
        str(CHECKPOINT_SIZE_MB),
        "--checkpoint-wire-scale",
        str(CHECKPOINT_WIRE_SCALE),
        "--dram-capacity-mb",
        str(DRAM_CAPACITY_MB),
        "--gpu-to-dram-bandwidth-mb-s",
        str(GPU_TO_DRAM_BANDWIDTH_MB_S),
        "--ssd-capacity-mb",
        str(SSD_CAPACITY_MB),
        "--ssd-bandwidth-mb-s",
        str(SSD_BANDWIDTH_MB_S),
        "--process-failure-percent-per-second",
        str(PROCESS_FAILURE_PERCENT_PER_SECOND),
        "--node-failure-percent-per-second",
        str(NODE_FAILURE_PERCENT_PER_SECOND),
        "--random-seed",
        str(RANDOM_SEED),
    ]


# =====================================================================
# PROCESS SUPERVISION AND RECOVERY
# =====================================================================


def terminate_processes(processes: list[tuple[dict[str, Any], Any]]) -> None:
    for _, process in processes:
        if process.poll() is None:
            process.terminate()

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if all(process.poll() is not None for _, process in processes):
            return
        time.sleep(0.05)

    for _, process in processes:
        if process.poll() is None:
            process.kill()


def load_failure_event(attempt_directory: Path) -> dict[str, Any] | None:
    event_files = sorted(attempt_directory.glob("rank-*-failure.json"))
    if not event_files:
        return None

    events = [json.loads(path.read_text(encoding="utf-8")) for path in event_files]
    events.sort(key=lambda event: str(event.get("wall_time_utc", "")))
    return events[0]


def monitor_attempt(
    processes: list[tuple[dict[str, Any], Any]],
    attempt_directory: Path,
) -> tuple[bool, dict[str, Any] | None]:
    while True:
        statuses = [process.poll() for _, process in processes]

        failed_indexes = [
            index
            for index, status in enumerate(statuses)
            if status is not None and status != 0
        ]
        if failed_indexes:
            # Give the failing worker a brief moment to atomically publish its
            # failure event before the supervisor reads it.
            time.sleep(0.15)
            event = load_failure_event(attempt_directory)
            if event is None:
                index = failed_indexes[0]
                configuration, process = processes[index]
                event = {
                    "type": "worker_failure",
                    "failure_type": "process",
                    "rank": configuration["rank"],
                    "host": configuration["name"],
                    "iteration": 0,
                    "stage": "unknown/peer-induced",
                    "exit_code": process.poll(),
                    "message": "Worker exited without a failure-event file",
                }
            return False, event

        if all(status == 0 for status in statuses):
            return True, None

        time.sleep(0.05)


def append_supervisor_event(event_file: Path, event: dict[str, Any]) -> None:
    event_file.parent.mkdir(parents=True, exist_ok=True)
    with event_file.open("a", encoding="utf-8") as file:
        file.write(json.dumps(event, separators=(",", ":")) + "\n")


def recover_after_failure(
    net: Mininet,
    failure: dict[str, Any],
    supervisor_event_file: Path,
) -> float:
    failure_type = str(failure.get("failure_type", "process"))
    failed_host = str(failure.get("host", ""))

    if failure_type == "node":
        delay = NODE_RESTART_TIME_SECONDS
        if failed_host:
            try:
                net.configLinkStatus(failed_host, "s1", "down")
            except Exception as error:  # noqa: BLE001
                print(f"*** Warning: could not bring {failed_host} link down: {error}")
    else:
        delay = PROCESS_RESTART_TIME_SECONDS

    print(
        f"\n*** {failure_type.upper()} FAILURE: "
        f"host={failure.get('host')}, rank={failure.get('rank')}, "
        f"iteration={failure.get('iteration')}, stage={failure.get('stage')}"
    )
    print(f"*** Restarting the complete distributed job after {delay:.3f}s\n", flush=True)

    recovery_start = time.perf_counter()
    time.sleep(delay)

    if failure_type == "node" and failed_host:
        try:
            net.configLinkStatus(failed_host, "s1", "up")
        except Exception as error:  # noqa: BLE001
            print(f"*** Warning: could not bring {failed_host} link up: {error}")

    actual_delay = time.perf_counter() - recovery_start
    append_supervisor_event(
        supervisor_event_file,
        {
            "type": "supervisor_restart",
            "failure_type": failure_type,
            "failed_host": failure.get("host"),
            "failed_rank": failure.get("rank"),
            "failure_iteration": failure.get("iteration"),
            "failure_stage": failure.get("stage"),
            "restart_delay_s": actual_delay,
            "wall_time_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    return actual_delay


# =====================================================================
# EXPERIMENT EXECUTION
# =====================================================================


def run_worker_variant(
    net: Mininet,
    mininet_hosts: dict[str, Any],
    host_configurations: list[dict[str, Any]],
    results_root: Path,
    run_id: str,
    variant: str,
) -> dict[str, Any]:
    worker_script = Path(__file__).with_name(WORKER_SCRIPTS[variant]).resolve()
    if not worker_script.exists():
        raise FileNotFoundError(f"Could not find {worker_script}")

    experiment = f"{run_id}-{variant}"
    variant_directory = results_root / variant
    variant_directory.mkdir(parents=True, exist_ok=True)
    supervisor_event_file = variant_directory / "supervisor-events.jsonl"

    print("\n" + "#" * 96)
    print(f"STARTING EXPERIMENT: {variant}")
    print(f"OBJECT-STORE NAMESPACE: {experiment}")
    print("#" * 96 + "\n", flush=True)

    wall_start = time.perf_counter()
    failures: list[dict[str, Any]] = []
    total_restart_delay_s = 0.0
    final_results: list[dict[str, Any]] = []

    for attempt in range(MAX_RECOVERY_ATTEMPTS + 1):
        attempt_directory = variant_directory / f"attempt-{attempt:02d}"
        attempt_directory.mkdir(parents=True, exist_ok=True)

        print(f"\n*** Launching attempt {attempt} for {variant}\n", flush=True)
        processes: list[tuple[dict[str, Any], Any]] = []

        for configuration in host_configurations:
            host = mininet_hosts[str(configuration["name"])]
            result_file = attempt_directory / f"rank-{configuration['rank']}-result.json"
            failure_event_file = (
                attempt_directory / f"rank-{configuration['rank']}-failure.json"
            )
            command = create_worker_command(
                worker_script=worker_script,
                experiment=experiment,
                variant=variant,
                attempt=attempt,
                host_configuration=configuration,
                result_file=result_file,
                failure_event_file=failure_event_file,
            )
            print(
                f"*** Starting {configuration['name']}: rank={configuration['rank']}, "
                f"ip={configuration['ip']}, attempt={attempt}",
                flush=True,
            )
            process = host.popen(command, stdout=None, stderr=None)
            processes.append((configuration, process))

        completed, failure = monitor_attempt(processes, attempt_directory)

        if completed:
            for configuration, _ in processes:
                result_file = (
                    attempt_directory
                    / f"rank-{configuration['rank']}-result.json"
                )
                if not result_file.exists():
                    raise FileNotFoundError(f"Missing result file: {result_file}")
                final_results.append(
                    json.loads(result_file.read_text(encoding="utf-8"))
                )
            break

        terminate_processes(processes)
        assert failure is not None
        failures.append(failure)
        total_restart_delay_s += recover_after_failure(
            net=net,
            failure=failure,
            supervisor_event_file=supervisor_event_file,
        )
    else:
        raise RuntimeError(
            f"{variant} exceeded MAX_RECOVERY_ATTEMPTS={MAX_RECOVERY_ATTEMPTS}"
        )

    wall_runtime_s = time.perf_counter() - wall_start
    final_results.sort(key=lambda result: int(result["rank"]))

    summary = aggregate_variant(
        experiment=experiment,
        variant=variant,
        wall_runtime_s=wall_runtime_s,
        failures=failures,
        total_restart_delay_s=total_restart_delay_s,
        final_results=final_results,
        object_store_event_file=results_root / "object-store-events.jsonl",
    )

    (variant_directory / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\n" + "#" * 96)
    print(f"COMPLETED EXPERIMENT: {variant}")
    print("#" * 96 + "\n", flush=True)
    return summary


# =====================================================================
# METRIC AGGREGATION
# =====================================================================


def read_json_lines(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def aggregate_variant(
    experiment: str,
    variant: str,
    wall_runtime_s: float,
    failures: list[dict[str, Any]],
    total_restart_delay_s: float,
    final_results: list[dict[str, Any]],
    object_store_event_file: Path,
) -> dict[str, Any]:
    store_events = [
        event
        for event in read_json_lines(object_store_event_file)
        if event.get("experiment") == experiment
    ]
    puts = [event for event in store_events if event.get("type") == "checkpoint_put"]
    gets = [event for event in store_events if event.get("type") == "recovery_get"]
    misses = [event for event in store_events if event.get("type") == "recovery_get_miss"]

    iteration_durations: list[float] = []
    gradient_sync_durations: list[float] = []
    final_attempt_recovery_s = 0.0
    gradient_bytes_sent = 0
    gradient_bytes_received = 0

    for result in final_results:
        final_attempt_recovery_s += float(result["counters"]["recovery_total_s"])
        gradient_bytes_sent += int(result["counters"]["gradient_bytes_sent"])
        gradient_bytes_received += int(result["counters"]["gradient_bytes_received"])
        iteration_durations.extend(
            float(record["iteration_s"]) for record in result["iterations"]
        )
        gradient_sync_durations.extend(
            float(record["gradient_sync_s"]) for record in result["iterations"]
        )

    latest_checkpoint_iteration = max(
        (int(event["iteration"]) for event in puts), default=0
    )
    process_failures = sum(
        1 for failure in failures if failure.get("failure_type") == "process"
    )
    node_failures = sum(
        1 for failure in failures if failure.get("failure_type") == "node"
    )

    return {
        "experiment": experiment,
        "variant": variant,
        "target_iterations": TARGET_ITERATIONS,
        "target_reached": all(
            int(result["last_completed_iteration"]) >= TARGET_ITERATIONS
            for result in final_results
        ),
        "wall_runtime_s": wall_runtime_s,
        "attempts": len(failures) + 1,
        "failures_total": len(failures),
        "process_failures": process_failures,
        "node_failures": node_failures,
        "restart_delay_s": total_restart_delay_s,
        "checkpoint_puts": len(puts),
        "latest_checkpoint_iteration": latest_checkpoint_iteration,
        "recovery_get_hits": len(gets),
        "recovery_get_misses": len(misses),
        "object_store_checkpoint_logical_gb": sum(
            int(event["logical_bytes"]) for event in puts
        )
        / 1024
        / 1024
        / 1024,
        "object_store_checkpoint_wire_mb": sum(
            int(event["wire_bytes"]) for event in puts
        )
        / 1024
        / 1024,
        "recovery_wire_mb": sum(int(event["wire_bytes"]) for event in gets)
        / 1024
        / 1024,
        "final_attempt_recovery_s_sum": final_attempt_recovery_s,
        "mean_final_attempt_iteration_s": (
            statistics.mean(iteration_durations) if iteration_durations else 0.0
        ),
        "mean_final_attempt_gradient_sync_s": (
            statistics.mean(gradient_sync_durations)
            if gradient_sync_durations
            else 0.0
        ),
        "gradient_bytes_sent": gradient_bytes_sent,
        "gradient_bytes_received": gradient_bytes_received,
        "effective_completed_iterations_per_s": TARGET_ITERATIONS / wall_runtime_s,
        "failures": failures,
    }


def write_combined_summary(results_root: Path, summaries: list[dict[str, Any]]) -> None:
    (results_root / "summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )

    csv_path = results_root / "summary.csv"
    fields = [
        "variant",
        "target_reached",
        "wall_runtime_s",
        "attempts",
        "failures_total",
        "process_failures",
        "node_failures",
        "restart_delay_s",
        "checkpoint_puts",
        "latest_checkpoint_iteration",
        "recovery_get_hits",
        "recovery_get_misses",
        "object_store_checkpoint_logical_gb",
        "object_store_checkpoint_wire_mb",
        "recovery_wire_mb",
        "effective_completed_iterations_per_s",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({field: summary[field] for field in fields})


def print_result_tables(summaries: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 96)
    print("FAILURE-AWARE EXPERIMENT SUMMARY")
    print("=" * 96)
    rows = []
    for summary in summaries:
        rows.append(
            [
                summary["variant"],
                summary["target_reached"],
                f"{summary['wall_runtime_s']:.3f}",
                summary["attempts"],
                summary["failures_total"],
                summary["process_failures"],
                summary["node_failures"],
                f"{summary['restart_delay_s']:.3f}",
                summary["latest_checkpoint_iteration"],
                f"{summary['effective_completed_iterations_per_s']:.4f}",
            ]
        )
    print_table(
        [
            "variant",
            "target",
            "runtime_s",
            "attempts",
            "failures",
            "proc",
            "node",
            "restart_s",
            "latest_ckpt",
            "iter/s",
        ],
        rows,
    )

    print("\nOBJECT-STORE AND RECOVERY SUMMARY")
    rows = []
    for summary in summaries:
        rows.append(
            [
                summary["variant"],
                summary["checkpoint_puts"],
                summary["recovery_get_hits"],
                summary["recovery_get_misses"],
                f"{summary['object_store_checkpoint_logical_gb']:.2f}",
                f"{summary['object_store_checkpoint_wire_mb']:.2f}",
                f"{summary['recovery_wire_mb']:.2f}",
            ]
        )
    print_table(
        [
            "variant",
            "puts",
            "get_hits",
            "get_misses",
            "logical_put_GB",
            "wire_put_MB",
            "recovery_MB",
        ],
        rows,
    )


# =====================================================================
# MAIN
# =====================================================================


def main() -> None:
    validate_configuration()
    print_configuration_summary()

    base_directory = Path(__file__).resolve().parent
    results_root = base_directory / RESULTS_DIRECTORY_NAME
    shutil.rmtree(results_root, ignore_errors=True)
    results_root.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    host_configurations = create_host_configurations()

    net = Mininet(
        controller=None,
        switch=OVSBridge,
        link=TCLink,
        autoSetMacs=True,
    )
    switch = net.addSwitch("s1")

    mininet_hosts: dict[str, Any] = {}
    for configuration in host_configurations:
        host = net.addHost(
            str(configuration["name"]),
            ip=f"{configuration['ip']}/24",
        )
        mininet_hosts[str(configuration["name"])] = host
        net.addLink(
            host,
            switch,
            bw=NETWORK_BANDWIDTH_MBPS,
            delay=f"{LINK_DELAY_MS}ms",
            use_tbf=True,
            latency_ms=100,
        )

    object_store_host = net.addHost("store1", ip=f"{OBJECT_STORE_IP}/24")
    net.addLink(
        object_store_host,
        switch,
        bw=STORE_NETWORK_BANDWIDTH_MBPS,
        delay=f"{LINK_DELAY_MS}ms",
        use_tbf=True,
        latency_ms=100,
    )

    object_store_process = None
    summaries: list[dict[str, Any]] = []

    try:
        net.start()
        print("*** Testing worker and object-store reachability", flush=True)
        packet_loss = net.pingAll()
        if packet_loss != 0:
            raise RuntimeError(f"Connectivity test failed: {packet_loss}% packet loss")

        object_store_script = base_directory / "object_store.py"
        ready_file = results_root / "object-store.ready"
        event_file = results_root / "object-store-events.jsonl"
        ready_file.unlink(missing_ok=True)

        object_store_process = object_store_host.popen(
            [
                "python3",
                "-u",
                str(object_store_script),
                "--host",
                OBJECT_STORE_IP,
                "--port",
                str(OBJECT_STORE_PORT),
                "--write-bandwidth-mb-s",
                str(OBJECT_STORE_WRITE_BANDWIDTH_MB_S),
                "--read-bandwidth-mb-s",
                str(OBJECT_STORE_READ_BANDWIDTH_MB_S),
                "--event-file",
                str(event_file),
                "--ready-file",
                str(ready_file),
            ],
            stdout=None,
            stderr=None,
        )

        ready_deadline = time.monotonic() + 10.0
        while not ready_file.exists():
            if object_store_process.poll() is not None:
                raise RuntimeError("Object-store process exited during startup")
            if time.monotonic() >= ready_deadline:
                raise TimeoutError("Object-store server did not become ready")
            time.sleep(0.05)

        for variant in WORKER_VARIANTS_TO_RUN:
            summaries.append(
                run_worker_variant(
                    net=net,
                    mininet_hosts=mininet_hosts,
                    host_configurations=host_configurations,
                    results_root=results_root,
                    run_id=run_id,
                    variant=variant,
                )
            )

        write_combined_summary(results_root, summaries)
        print_result_tables(summaries)
        print(f"\nResults written to: {results_root}\n", flush=True)

    finally:
        if object_store_process is not None and object_store_process.poll() is None:
            object_store_process.terminate()
            try:
                object_store_process.wait(timeout=2.0)
            except Exception:
                object_store_process.kill()
        net.stop()


if __name__ == "__main__":
    setLogLevel("info")
    main()
