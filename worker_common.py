#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from distributed_training import DistributedTraining, SimulatedFailure


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--result-file", type=Path, required=True)
    parser.add_argument("--failure-event-file", type=Path, required=True)

    parser.add_argument("--name", required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--local-ip", required=True)
    parser.add_argument("--master-ip", required=True)
    parser.add_argument("--master-port", type=int, required=True)
    parser.add_argument("--object-store-ip", required=True)
    parser.add_argument("--object-store-port", type=int, required=True)

    parser.add_argument("--network-bandwidth-mbps", type=float, required=True)
    parser.add_argument("--target-iterations", type=int, required=True)
    parser.add_argument("--forward-time", type=float, required=True)
    parser.add_argument("--backward-time", type=float, required=True)
    parser.add_argument("--update-time", type=float, required=True)
    parser.add_argument("--gradient-size-mb", type=float, required=True)

    parser.add_argument("--checkpoint-interval", type=int, required=True)
    parser.add_argument("--checkpoint-size-mb", type=float, required=True)
    parser.add_argument("--checkpoint-wire-scale", type=float, required=True)
    parser.add_argument("--dram-capacity-mb", type=float, required=True)
    parser.add_argument("--gpu-to-dram-bandwidth-mb-s", type=float, required=True)

    # Kept for the local SSD path, even though the current checkpoint worker
    # sends checkpoints to the external object-store host.
    parser.add_argument("--ssd-capacity-mb", type=float, required=True)
    parser.add_argument("--ssd-bandwidth-mb-s", type=float, required=True)

    parser.add_argument(
        "--process-failure-percent-per-second", type=float, required=True
    )
    parser.add_argument(
        "--node-failure-percent-per-second", type=float, required=True
    )
    parser.add_argument("--random-seed", type=int, required=True)


def validate_common_arguments(args: argparse.Namespace) -> None:
    if args.target_iterations < 1:
        raise ValueError("target-iterations must be at least 1")
    if args.checkpoint_interval < 1:
        raise ValueError("checkpoint-interval must be at least 1")
    if args.attempt < 0:
        raise ValueError("attempt cannot be negative")


def build_training(args: argparse.Namespace) -> DistributedTraining:
    validate_common_arguments(args)
    return DistributedTraining(
        experiment=args.experiment,
        variant=args.variant,
        attempt=args.attempt,
        name=args.name,
        rank=args.rank,
        world_size=args.world_size,
        local_ip=args.local_ip,
        master_ip=args.master_ip,
        master_port=args.master_port,
        object_store_ip=args.object_store_ip,
        object_store_port=args.object_store_port,
        network_bandwidth_mbps=args.network_bandwidth_mbps,
        forward_time=args.forward_time,
        backward_time=args.backward_time,
        update_time=args.update_time,
        gradient_size_mb=args.gradient_size_mb,
        checkpoint_size_mb=args.checkpoint_size_mb,
        checkpoint_wire_scale=args.checkpoint_wire_scale,
        dram_capacity_mb=args.dram_capacity_mb,
        gpu_to_dram_bandwidth_mb_s=args.gpu_to_dram_bandwidth_mb_s,
        ssd_capacity_mb=args.ssd_capacity_mb,
        ssd_bandwidth_mb_s=args.ssd_bandwidth_mb_s,
        process_failure_percent_per_second=(
            args.process_failure_percent_per_second
        ),
        node_failure_percent_per_second=args.node_failure_percent_per_second,
        random_seed=args.random_seed,
        failure_event_file=args.failure_event_file,
    )


def run_worker(worker: object, training: DistributedTraining) -> None:
    try:
        worker.run()  # type: ignore[attr-defined]
    except SimulatedFailure as failure:
        training.write_failure_event(failure)
        sys.exit(failure.exit_code)
