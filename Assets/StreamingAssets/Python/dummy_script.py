"""Dummy script called from Unity.

Reads a JSON object from stdin, e.g. {"values": [1.0, 2.5, 4.0]},
and prints a JSON object with a few statistics to stdout.
Errors go to stderr with a non-zero exit code, so Unity can detect them.
"""
import json
import sys


def main():
    raw = sys.stdin.read()
    data = json.loads(raw) if raw.strip() else {}
    values = [float(v) for v in data.get("values", [])]

    result = {
        "count": len(values),
        "sum": sum(values),
        "mean": sum(values) / len(values) if values else 0.0,
        "message": f"Hello from Python {sys.version.split()[0]}",
    }
    print(json.dumps(result))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # report any failure to Unity through stderr
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)