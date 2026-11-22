"""Receives session data from Unity (JSON on stdin) and appends it to two CSV tables:

    <output_dir>/<session_id>_trajectories.csv   one row per (episode_id, step_id)
    <output_dir>/<session_id>_answers.csv        one row per (questionnaire_id, question, answer)

Prints a JSON summary on stdout. Errors go to stderr with exit code 1.
"""
import json
import os
import sys

import pandas as pd

TRAJECTORY_COLUMNS = [
    "episode_id", "step_id", "time",
    "pos_x", "pos_y", "pos_z", "vel_x", "vel_z",
    "action_x", "action_z", "score", "cumulative_reward",
]
ANSWER_COLUMNS = ["questionnaire_id", "question_index", "question", "answer", "timestamp"]


def append_table(rows, columns, path):
    """Append rows to a CSV, writing the header only when the file is new."""
    if not rows:
        return 0
    df = pd.DataFrame(rows).reindex(columns=columns)
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)
    return len(df)


def main():
    raw = sys.stdin.buffer.read().decode("utf-8-sig")   # tolerate a byte-order mark
    data = json.loads(raw)

    output_dir = data.get("output_dir") or "."
    session_id = data.get("session_id") or "session"
    os.makedirs(output_dir, exist_ok=True)

    trajectories_path = os.path.join(output_dir, f"{session_id}_trajectories.csv")
    answers_path = os.path.join(output_dir, f"{session_id}_answers.csv")

    n_traj = append_table(data.get("trajectories", []), TRAJECTORY_COLUMNS, trajectories_path)
    n_ans = append_table(data.get("answers", []), ANSWER_COLUMNS, answers_path)

    print(json.dumps({
        "ok": True,
        "n_trajectory_rows": n_traj,
        "n_answer_rows": n_ans,
        "trajectories_path": trajectories_path,
        "answers_path": answers_path,
    }))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)