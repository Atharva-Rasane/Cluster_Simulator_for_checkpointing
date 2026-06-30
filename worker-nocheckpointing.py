#!/usr/bin/env python3

from __future__ import annotations

import argparse

from worker_common import add_common_arguments, build_training, run_worker


class NoCheckpointWorker:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.training = build_training(args)

    def run(self) -> None:
        self.training.log_configuration(
            target_iterations=self.args.target_iterations,
            checkpoint_interval=self.args.checkpoint_interval,
        )
        self.training.log(
            None,
            "WORKER",
            "START",
            "baseline without checkpoint creation",
        )

        # Every restart still asks the object store for the latest checkpoint.
        # For this baseline the response will normally be NOT_FOUND, causing the
        # distributed job to restart from iteration 1.
        checkpoint_iteration = self.training.recover_from_object_store()
        next_iteration = checkpoint_iteration + 1

        for iteration in range(next_iteration, self.args.target_iterations + 1):
            self.training.begin_iteration(iteration, self.args.target_iterations)
            self.training.training_stage(iteration)
            self.training.update_stage(iteration)
            self.training.log(
                iteration,
                "CHECKPOINT",
                "SKIP",
                "worker variant disables checkpoint creation",
            )
            self.training.end_iteration(iteration, self.args.target_iterations)

        self.training.log(
            None,
            "WORKER",
            "TARGET",
            f"reached_iteration={self.args.target_iterations}",
        )
        self.training.write_result(
            result_file=self.args.result_file,
            target_iterations=self.args.target_iterations,
            checkpoint_interval=self.args.checkpoint_interval,
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    worker = NoCheckpointWorker(args)
    run_worker(worker, worker.training)


if __name__ == "__main__":
    main()
