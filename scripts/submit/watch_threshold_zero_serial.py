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
OUT = ROOT / "outputs/threshold_zero_all_20260918"
STATE = OUT / "serial-event-state.json"
EVENT_ID = "auroralf-cp6-threshold-zero-11690770-20260918"
JOB = "11690770"


def write(value):
    temporary = STATE.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(STATE)


def query():
    command = "sacct -X -n -P -j 11690770 --format=JobID,Cluster,Submit,User,State%40,ExitCode"
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
        if (
            row["cluster"] != "hpc5"
            or row["user"] != "xuelei"
            or row["submitted"] != "2026-09-18T10:59:12"
        ):
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
            followup="Continue the authorized mu=0 ALL-result recalculation: HeII, PISN, target comparisons, SFRD, reionization and main slides. Inspect actual outputs, follow outputs/threshold_zero_all_20260918/FOLLOWUP.md. The single-node sequential job includes HeII targets/uniform populations, SFRD and eight ionizing-source shards. The same job also merges the source table; validate that product, then rerun full spatial reionization on the available local node3 GPU using SmallScale21cm. Finish all numerical text/plots, compile, visually review and sync Zotero. Do not stop after merely fetching data. Preserve unrelated edits and report only meaningful changes.",
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thread", required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
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
