import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


def _safe_scalars(acc, tag):
    try:
        return acc.Scalars(tag)
    except Exception:
        return []


def _estimate_runtime_seconds(acc):
    """Estimate runtime from earliest to latest scalar wall_time in the run."""
    tags = acc.Tags().get("scalars", [])
    min_t = None
    max_t = None
    for tag in tags:
        events = _safe_scalars(acc, tag)
        if not events:
            continue
        t0 = events[0].wall_time
        t1 = events[-1].wall_time
        min_t = t0 if min_t is None else min(min_t, t0)
        max_t = t1 if max_t is None else max(max_t, t1)

    if min_t is None or max_t is None:
        return None
    return max(0.0, max_t - min_t)


def _extract_run_metrics(run_dir):
    acc = event_accumulator.EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    acc.Reload()

    run_name = run_dir.name

    run_write_time = os.path.getmtime(run_dir)

    # Preferred explicit run summary (new runs)
    final_score_events = _safe_scalars(acc, "Run/final_match_score")
    runtime_events = _safe_scalars(acc, "Run/runtime_seconds")

    if final_score_events:
        final_score = float(final_score_events[-1].value)
    else:
        # Backward-compatible fallback for older runs: sum all segment rewards.
        seg_events = _safe_scalars(acc, "Segment/segment_reward")
        final_score = float(sum(e.value for e in seg_events)) if seg_events else None

    if runtime_events:
        runtime_seconds = float(runtime_events[-1].value)
    else:
        runtime_seconds = _estimate_runtime_seconds(acc)

    return {
        "run_name": run_name,
        "runtime_seconds": runtime_seconds,
        "final_score": final_score,
        "run_write_time": run_write_time,
    }


def collect_run_metrics(runs_dir):
    runs_path = Path(runs_dir)
    run_dirs = [p for p in runs_path.iterdir() if p.is_dir()]
    rows = []

    for run_dir in sorted(run_dirs):
        try:
            metrics = _extract_run_metrics(run_dir)
        except Exception:
            continue

        if metrics["runtime_seconds"] is None or metrics["final_score"] is None:
            continue
        rows.append(metrics)

    return rows


def export_csv(rows, output_csv):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["run_name", "runtime_seconds", "final_score", "run_write_time"])
        writer.writeheader()
        writer.writerows(rows)


def export_plot(rows, output_png, title):
    output_png.parent.mkdir(parents=True, exist_ok=True)

    x = [r["runtime_seconds"] for r in rows]
    y = [r["final_score"] for r in rows]

    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, alpha=0.8)
    plt.xlabel("Runtime (seconds)")
    plt.ylabel("Final Score")
    plt.title(title)
    plt.grid(True, alpha=0.3)

    # Optional trend line when we have enough points.
    if len(rows) >= 2:
        import numpy as np

        coeffs = np.polyfit(x, y, 1)
        trend = np.poly1d(coeffs)
        xs = np.linspace(min(x), max(x), 100)
        plt.plot(xs, trend(xs), linestyle="--", linewidth=1.5)

    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    plt.close()


def _normalize_optional_labels(raw_labels):
    """Split comma-separated labels; empty tokens become None."""
    if raw_labels is None:
        return []
    return [token.strip() or None for token in raw_labels.split(",")]


def _normalize_label_value(value):
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    return str(value).strip() or None


def _load_split_labels_from_json(json_path):
    """Load labels from JSON list or object with a split_labels key."""
    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict):
        payload = payload.get("split_labels", [])

    if not isinstance(payload, list):
        raise ValueError("Split labels JSON must be a list or an object with a 'split_labels' list.")

    return [_normalize_label_value(item) for item in payload]


def export_progress_plot(rows, output_png, title, rolling_window, splits, split_labels=None):
    output_png.parent.mkdir(parents=True, exist_ok=True)

    x = list(range(1, len(rows) + 1))
    y = [r["final_score"] for r in rows]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, y, marker="o", linestyle="", alpha=0.6, label="Run score")

    if len(rows) >= 2:
        import numpy as np

        window = max(2, min(int(rolling_window), len(y)))
        kernel = np.ones(window) / window
        smooth = np.convolve(np.array(y, dtype=float), kernel, mode="valid")
        smooth_x = list(range(window, len(y) + 1))
        ax.plot(smooth_x, smooth, linewidth=2.0, label=f"Rolling avg ({window})")

    labels = split_labels or []
    for i, split_start_index in enumerate(splits):
        # Draw at boundary between the previous run and this split-start run.
        x_pos = max(0.5, float(split_start_index) - 0.5)
        provided_name = labels[i] if i < len(labels) else None
        line_label = f"Split: {provided_name}" if provided_name else ("Split boundary" if i == 0 else None)
        ax.axvline(x=x_pos, linestyle="--", linewidth=1.2, alpha=0.8, color="tab:red", label=line_label)

        if provided_name:
            y_max = ax.get_ylim()[1]
            ax.text(
                x_pos + 0.03,
                y_max,
                provided_name,
                rotation=90,
                va="top",
                ha="left",
                fontsize=8,
                color="tab:red",
            )

    ax.set_xlabel("Run index (time-ordered)")
    ax.set_ylabel("Final Score")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    # Only show the top 2 lines in the legend to avoid clutter (the split lines will be labeled but not all shown).
    h,l = ax.get_legend_handles_labels()
    ax.legend(loc="upper left", fontsize=8, handles=h[:2], labels=l[:2])
    fig.tight_layout()
    fig.savefig(output_png, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Export a scatter plot of run final score vs runtime from TensorBoard run folders."
    )
    parser.add_argument(
        "--runs-dir",
        default=str(Path(__file__).resolve().parents[1] / "runs"),
        help="Path to runs directory (default: Main/runs)",
    )
    parser.add_argument(
        "--output-png",
        default=str(Path(__file__).resolve().parent / "run_score_vs_runtime.png"),
        help="Output PNG path",
    )
    parser.add_argument(
        "--output-csv",
        default=str(Path(__file__).resolve().parent / "run_score_vs_runtime.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--output-progress-png",
        default=str(Path(__file__).resolve().parent / "run_score_over_time.png"),
        help="Output PNG path for score-vs-run-order trend",
    )
    parser.add_argument(
        "--title",
        default="Run Final Score vs Runtime",
        help="Plot title",
    )
    parser.add_argument(
        "--progress-title",
        default="Run Final Score Over Time",
        help="Title for score-vs-run-order trend plot",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=10,
        help="Rolling average window size for the trend plot",
    )
    parser.add_argument(
        "--split-labels",
        default="",
        help=(
            "Comma-separated labels for detected split lines in order. "
            "Empty entries are treated as None (example: 'Phase 2,,Final push')."
        ),
    )
    parser.add_argument(
        "--split-labels-json",
        default="splitLabels.json",
        help=(
            "Path to JSON file containing split labels. "
            "Supported formats: [\"A\", null, \"B\"] or {\"split_labels\": [...]}."
        ),
    )
    args = parser.parse_args()

    rows = collect_run_metrics(args.runs_dir)
    if not rows:
        raise SystemExit("No usable run metrics found. Ensure runs contain TensorBoard scalar events.")

    rows.sort(key=lambda r: r["run_write_time"])

    splits = []
    prevRunTime = None
    # Extract split boundaries: index where a new run segment starts.
    for idx, r in enumerate(rows, start=1):
        if prevRunTime is None:
            prevRunTime = r["run_write_time"]
        else:
            # Check to see if greater than an hour passed inbetween runs, if so we can consider it a split
            if r["run_write_time"] - prevRunTime > 3600:
                splits.append(idx)
            prevRunTime = r["run_write_time"]

    split_labels_json = args.split_labels_json.strip()
    if split_labels_json:
        split_labels_path = Path(split_labels_json)
        if not split_labels_path.is_absolute():
            split_labels_path = Path(__file__).resolve().parent / split_labels_path
        if not split_labels_path.exists():
            raise SystemExit(f"Split labels JSON not found: {split_labels_path}")

        try:
            split_labels = _load_split_labels_from_json(split_labels_path)
        except Exception as exc:
            raise SystemExit(f"Failed to parse split labels JSON '{split_labels_path}': {exc}")
    else:
        split_labels = _normalize_optional_labels(args.split_labels)

    if len(split_labels) < len(splits):
        split_labels.extend([None] * (len(splits) - len(split_labels)))
    else:
        split_labels = split_labels[: len(splits)]
        
            
    export_csv(rows, Path(args.output_csv))
    export_plot(rows, Path(args.output_png), args.title)
    export_progress_plot(
        rows,
        Path(args.output_progress_png),
        args.progress_title,
        args.rolling_window,
        splits,
        split_labels,
    )

    print(f"Exported {len(rows)} runs")
    print(f"CSV: {args.output_csv}")
    print(f"Plot: {args.output_png}")
    print(f"Trend plot: {args.output_progress_png}")


if __name__ == "__main__":
    main()
