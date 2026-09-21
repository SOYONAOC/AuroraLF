"""Remote cp6 event adapter for this campaign; one wakeup in its owning task.

Uses the installed job-watch AppServer transport. Polls only the authorized
single job over SSH; contacts Codex only on completion/failure and waits for idle.
An uncertain delivery is held for reconciliation, never blindly resent.
"""

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

WATCH_ROOT = Path.home() / ".local/share/codex-job-watch"
sys.path.insert(0, str(WATCH_ROOT))
from watch import DEFAULT_SOCKET, TERMINAL, AppServer, RpcError  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/pisn_lowz_20260918"
STATE = OUT / "serial-event-state.json"
EVENT_ID = "auroralf-cp6-pisn-lowz-11693155-20260918"
JOB = "11693155"
SUBMITTED = "2026-09-18T15:54:10"
FOLLOWUP = "Continue the authorized low-redshift PISN calculation. Job 11693155 is a single 56-CPU cp6 allocation. Fetch and verify data_save/pisn_lowz_20260918 and outputs logs from sc:/fs2/home/xuelei/zhuhr/AuroraLF/releases/pisn-lowz-20260918-01/. Read docs/pisn-lowz-extrapolation.md. Validate all 15 cases, integration/time/mass convergence, MC uncertainty and overlap with previous z12.5/14.5 predictions; inspect mass-edge fractions. Diagnose any failure without changing physical parameters. Produce an independent labelled diagnostic plot of z1/2/3 rates plus high-z anchors, and compare the scope with Moriya2021 HSC luminous-template bound; do not interpret all-PISN rate as HSC detections or a formal exclusion. This model lacks enrichment/pristine survival. Keep main slides unchanged for now. Summarize numerical results, whether a like-for-like observational comparison is possible, and missing selection inputs. Preserve unrelated work."


def write(value):
    temporary = STATE.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(STATE)


def query():
    command = f"sacct -X -n -P -j {JOB} --format=JobID,Cluster,Submit,User,State%40,ExitCode"
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "sc", command],
        check=True,
        capture_output=True,
        text=True,
        timeout=45,
    )
    rows = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        fields = line.rstrip("|").split("|")
        if len(fields) != 6:
            raise ValueError("Unexpected accounting schema")
        row = dict(
            zip(("job", "cluster", "submitted", "user", "state", "exit_code"), fields, strict=True)
        )
        if row["cluster"] != "hpc5" or row["user"] != "xuelei" or row["submitted"] != SUBMITTED:
            raise ValueError("Remote scheduler identity mismatch")
        rows.append(row)
    if len(rows) != 1 or rows[0]["job"] != JOB:
        raise ValueError("Expected the single sequential job")
    return rows


def deliver(state, thread):
    client = AppServer(DEFAULT_SOCKET)
    try:
        if state["phase"] in ("dispatching", "uncertain"):
            history = client.call("thread/read", {"threadId": thread, "includeTurns": True})[
                "thread"
            ]
            matches = [t for t in history.get("turns", []) if EVENT_ID in json.dumps(t)]
            if matches:
                state.update(phase="delivered", turn_id=matches[-1]["id"])
                write(state)
                return True
            raise RuntimeError("Uncertain delivery held; manual reconciliation required")
        current = client.call("thread/read", {"threadId": thread, "includeTurns": False})["thread"]
        if Path(current["cwd"]).resolve() != ROOT:
            raise ValueError("Owning task cwd mismatch")
        if current["status"]["type"] == "active":
            return False
        if current["status"]["type"] == "notLoaded":
            current = client.call("thread/resume", {"threadId": thread})["thread"]
        if current["status"]["type"] != "idle":
            return False
        state["phase"] = "dispatching"
        write(state)
        payload = dict(
            event_id=EVENT_ID,
            source="Authorized remote cp6 campaign; single job identity checked",
            accounting=state["accounting"],
            followup=FOLLOWUP,
        )
        try:
            result = client.call(
                "turn/start",
                {
                    "threadId": thread,
                    "input": [],
                    "toolOutput": {
                        "name": "slurm_job_event",
                        "output": json.dumps(payload, ensure_ascii=False),
                    },
                },
            )
        except RpcError:
            state["phase"] = "waiting"
            write(state)
            raise
        except Exception:
            state["phase"] = "uncertain"
            write(state)
            raise
        state.update(phase="delivered", turn_id=result["turn"]["id"])
        write(state)
        return True
    finally:
        client.close()


def main():
    global OUT, STATE, EVENT_ID, JOB, SUBMITTED, FOLLOWUP
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thread", required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--tail", action="store_true")
    args = parser.parse_args()
    if args.tail:
        OUT = ROOT / "outputs/pisn_lowz_20260918/tail_events"
        OUT.mkdir(parents=True, exist_ok=True)
        STATE = OUT / "serial-event-state.json"
        JOB = "11693767"
        SUBMITTED = "2026-09-18T16:37:20"
        EVENT_ID = "auroralf-cp6-pisn-lowz-tail-11693767-20260918"
        FOLLOWUP = "Continue the authorized low-redshift PISN result validation. Job 11693767 checks missing high-mass halos on one 56-CPU cp6 node. Fetch data_save/pisn_lowz_mass_tail_20260918 and logs from sc:/fs2/home/xuelei/zhuhr/AuroraLF/releases/pisn-lowz-tail-20260918-01/. Read docs/pisn-lowz-extrapolation.md. Baseline job11693155 already verified 15 cases; z1/2/3 rates14.41/104.17/354.45 events per source year per cGpc^3 only cover halo masses1e5-1e12. Their highest mass decade contributes65/48/27%, so the tail1e12-1e15 must be included before claiming a complete mass integral. Verify all9 tail cases and hashes, numerical convergence and high-mass tail saturation; diagnose any failure without arbitrary parameter changes. Use scripts/plot/plot_pisn_lowz.py --tail data_save/pisn_lowz_mass_tail_20260918 to make independent diagnostic plots, inspect them, update docs with combined rates and limitations, and report to the user. Keep slides unchanged. Model lacks enrichment/pristine survival and predicts all PISNe; do not equate to HSC luminous-template counts or claim a formal exclusion. Do not repeatedly submit more batches absent a concrete unresolved concern."

    os.umask(0o077)
    with (OUT / "serial-event.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = (
            json.loads(STATE.read_text())
            if STATE.exists()
            else dict(phase="watching", thread=args.thread)
        )
        if state["thread"] != args.thread:
            raise ValueError("Owner changed")
        while state["phase"] != "delivered":
            try:
                rows = query()
                state.update(accounting=rows, checked_unix=time.time())
                terminal = [r for r in rows if r["state"].split()[0].rstrip("+") in TERMINAL]
                failed = [
                    r for r in terminal if r["state"] != "COMPLETED" or r["exit_code"] != "0:0"
                ]
                if (failed or len(terminal) == 1) and state["phase"] == "watching":
                    state["phase"] = "waiting"
                write(state)
                if args.check_only:
                    print(json.dumps(state, ensure_ascii=False))
                    return
                if state["phase"] != "watching" and deliver(state, args.thread):
                    print("Terminal event delivered to owning task", flush=True)
                    return
            except Exception as error:
                print(f"Event monitor error: {type(error).__name__}: {error}", flush=True)
                if args.check_only:
                    raise
            time.sleep(120)


if __name__ == "__main__":
    main()
