"""Use the DMDE submitter with explicit mail flags for older SLURM versions."""

import argparse
import importlib.util
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(add_help=False)
    allocation = parser.add_mutually_exclusive_group()
    allocation.add_argument("--fixed-cpus", type=int)
    allocation.add_argument("--leave-idle-cpus", type=int)
    options, remaining = parser.parse_known_args()
    if options.fixed_cpus is not None and options.fixed_cpus < 1:
        raise ValueError("--fixed-cpus must be positive")
    if options.leave_idle_cpus is not None and options.leave_idle_cpus < 1:
        raise ValueError("--leave-idle-cpus must be positive")
    sys.argv[1:] = remaining
    path = Path("/home/zhuhourui/.codex/skills/dmde-compute/scripts/submit_python_job.py")
    spec = importlib.util.spec_from_file_location("uvlf_dmde_submitter", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load scheduler helper: {path}")
    submitter = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = submitter
    spec.loader.exec_module(submitter)
    if options.fixed_cpus is not None:
        build_candidates = submitter.build_candidates

        def fixed_candidates(*args, **kwargs):
            candidates = build_candidates(*args, **kwargs)
            for item in candidates:
                eligible = (
                    item.usable_cpus > 0
                    and item.idle_cpus >= options.fixed_cpus
                    and item.total_cpus > options.fixed_cpus
                )
                item.usable_cpus = options.fixed_cpus if eligible else 0
                item.exclusive = False
                item.resource_score = item.usable_cpus * item.benchmark
            return candidates

        submitter.build_candidates = fixed_candidates
    if options.leave_idle_cpus is not None:
        build_candidates = submitter.build_candidates

        def available_candidates(table, *args, **kwargs):
            # User requested immediate shared placement with spare idle CPUs.
            candidates = build_candidates(
                table, cpu_fraction=1.0, min_usable_cpus=1, node_allocation="shared"
            )
            for item in candidates:
                item.usable_cpus = (
                    max(0, item.idle_cpus - options.leave_idle_cpus)
                    if item.usable_cpus > 0 and "debug" not in item.partition.lower()
                    else 0
                )
                item.exclusive = False
                item.resource_score = item.usable_cpus * item.benchmark
            return candidates

        original_rank = submitter.ranked_usable_candidates

        def available_rank(candidates):
            # Independent histories benefit from aggregate available throughput.
            return sorted(
                original_rank(candidates),
                key=lambda item: (item.resource_score, item.benchmark),
                reverse=True,
            )

        submitter.build_candidates = available_candidates
        submitter.ranked_usable_candidates = available_rank
    original = submitter.build_sbatch_command

    def with_mail(*args, **kwargs):
        command = original(*args, **kwargs)
        return [
            command[0],
            "--mail-user=lighmisamisa@agent.qq.com",
            "--mail-type=END,FAIL",
            *command[1:],
        ]

    submitter.build_sbatch_command = with_mail
    submitter.main()


if __name__ == "__main__":
    main()
