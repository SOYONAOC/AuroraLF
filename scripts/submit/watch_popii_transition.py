"""Event-only SSH monitor: continue this task after both cp6 jobs end or one fails."""

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path.home() / ".local/share/codex-job-watch"))
from watch import DEFAULT_SOCKET, TERMINAL, AppServer, RpcError  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/popii_transition_20260920"
STATE = OUT / "event-state.json"
EVENT = "auroralf-popii-transition-11713297-11713298-20260920"
IDENTITIES = {"11713297": "2026-09-20T19:18:00", "11713298": "2026-09-20T19:18:01"}


def write(state):
    p = STATE.with_suffix(".tmp")
    p.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")
    p.replace(STATE)


def query():
    command = (
        "sacct -X -n -P -j "
        + ",".join(IDENTITIES)
        + " --format=JobID,Cluster,Submit,User,State%40,ExitCode"
    )
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "sc", command],
        capture_output=True,
        text=True,
        timeout=45,
        check=True,
    )
    rows = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        fields = line.rstrip("|").split("|")
        row = dict(
            zip(("job", "cluster", "submitted", "user", "state", "exit_code"), fields, strict=True)
        )
        if (
            row["job"] not in IDENTITIES
            or row["cluster"] != "hpc5"
            or row["user"] != "xuelei"
            or row["submitted"] != IDENTITIES[row["job"]]
        ):
            raise ValueError("Scheduler identity mismatch")
        rows.append(row)
    if {r["job"] for r in rows} != set(IDENTITIES):
        raise ValueError("Missing job accounting")
    return rows


def deliver(state, thread):
    client = AppServer(DEFAULT_SOCKET)
    try:
        if state["phase"] in ("dispatching", "uncertain"):
            history = client.call("thread/read", {"threadId": thread, "includeTurns": True})[
                "thread"
            ]
            matches = [t for t in history.get("turns", []) if EVENT in json.dumps(t)]
            if matches:
                state.update(phase="delivered", turn_id=matches[-1]["id"])
                write(state)
                return True
            raise RuntimeError("Uncertain event held for reconciliation")
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
            event_id=EVENT,
            accounting=state["accounting"],
            followup="Continue the user-authorized PopIII->PopII UVLF and reionization comparison. Follow outputs/popii_transition_20260920/FOLLOWUP.md. Fetch and verify frozen cp6 results, check baseline and transition integration/censoring diagnostics, prepare both delayed source tables at common fesc=.064 and submit full spatial evolutions on actually available local GPUs. Preserve the baseline reference and all source parameters. Finish UVLF+neutral fraction+tau comparison slides, visual QA and Zotero sync. Do not stop after fetching source tables or merely reporting scheduler success. No messages for unchanged running state.",
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--thread", required=True)
    p.add_argument("--check-only", action="store_true")
    a = p.parse_args()
    os.umask(0o077)
    with (OUT / "event.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = (
            json.loads(STATE.read_text())
            if STATE.exists()
            else dict(phase="watching", thread=a.thread)
        )
        if state["thread"] != a.thread:
            raise ValueError("Owner changed")
        while state["phase"] != "delivered":
            try:
                rows = query()
                state.update(accounting=rows, checked_unix=time.time())
                terminal = [r for r in rows if r["state"].split()[0].rstrip("+") in TERMINAL]
                failed = [
                    r for r in terminal if r["state"] != "COMPLETED" or r["exit_code"] != "0:0"
                ]
                if (failed or len(terminal) == len(IDENTITIES)) and state["phase"] == "watching":
                    state["phase"] = "waiting"
                write(state)
                if a.check_only:
                    print(json.dumps(state, ensure_ascii=False))
                    return
                if state["phase"] != "watching" and deliver(state, a.thread):
                    return
            except Exception as error:
                print(f"Monitor error: {type(error).__name__}: {error}", flush=True)
                if a.check_only:
                    raise
            time.sleep(120)


if __name__ == "__main__":
    main()
