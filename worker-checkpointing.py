#!/usr/bin/env python3

from __future__ import annotations

import argparse

from worker_common import add_common_arguments, build_training, run_worker


class CheckpointWorker:
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
            "rank-0 synchronous checkpointing to external object store",
        )

        checkpoint_iteration = self.training.recover_from_object_store()
        next_iteration = checkpoint_iteration + 1

        for iteration in range(next_iteration, self.args.target_iterations + 1):
            self.training.begin_iteration(iteration, self.args.target_iterations)
            self.training.training_stage(iteration)
            self.training.update_stage(iteration)
            
            checkpoint_due = iteration % self.args.checkpoint_interval == 0
            if self.args.rank == 0 and checkpoint_due:
                self.training.checkpoint_to_object_store(iteration)
            elif self.args.rank != 0:
                self.training.log(
                    iteration,
                    "CHECKPOINT",
                    "SKIP",
                    "only rank 0 checkpoints",
                )
            else:
                self.training.log(
                    iteration,
                    "CHECKPOINT",
                    "SKIP",
                    f"interval={self.args.checkpoint_interval} not reached",
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
    worker = CheckpointWorker(args)
    run_worker(worker, worker.training)


if __name__ == "__main__":
    main()
