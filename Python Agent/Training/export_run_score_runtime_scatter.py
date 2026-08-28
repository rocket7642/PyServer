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
    success_events = _safe_scalars(acc, "Run/success")

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

    if success_events:
        run_success = float(success_events[-1].value)
    else:
        run_success = None

    return {
        "run_name": run_name,
        "runtime_seconds": runtime_seconds,
        "final_score": final_score,
        "run_write_time": run_write_time,
        "run_success": run_success,
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


def export_csv(rows, output_csv, failure_threshold):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    failure_flags, _ = _get_failure_flags(rows, failure_threshold)

    csv_rows = []
    for row, is_failure in zip(rows, failure_flags):
        csv_row = dict(row)
        csv_row["is_failure"] = int(is_failure)
        csv_rows.append(csv_row)

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["run_name", "runtime_seconds", "final_score", "run_write_time", "run_success", "is_failure"],
        )
        writer.writeheader()
        writer.writerows(csv_rows)


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


def _draw_split_lines(ax, splits, split_labels):
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


def _get_failure_flags(rows, failure_threshold):
    flags = []
    fallback_count = 0

    for r in rows:
        success_value = r.get("run_success")
        if success_value is None:
            fallback_count += 1
            flags.append(1 if float(r["final_score"]) < failure_threshold else 0)
        else:
            flags.append(0 if float(success_value) >= 0.5 else 1)

    return flags, fallback_count


def _draw_score_progress(ax, rows, rolling_window, splits, split_labels):
    x = list(range(1, len(rows) + 1))
    y = [r["final_score"] for r in rows]

    ax.plot(x, y, marker="o", linestyle="", alpha=0.6, label="Run score")

    if len(rows) >= 2:
        import numpy as np

        window = max(2, min(int(rolling_window), len(y)))
        y_arr = np.array(y, dtype=float)

        kernel = np.ones(window) / window
        rolling_avg = np.convolve(y_arr, kernel, mode="valid")
        rolling_x = list(range(window, len(y) + 1))
        ax.plot(rolling_x, rolling_avg, linewidth=1.8, label=f"Rolling avg ({window})")

        rolling_median = [float(np.median(y_arr[i - window : i])) for i in range(window, len(y) + 1)]
        q25 = [float(np.percentile(y_arr[i - window : i], 25)) for i in range(window, len(y) + 1)]
        q75 = [float(np.percentile(y_arr[i - window : i], 75)) for i in range(window, len(y) + 1)]

        ax.plot(rolling_x, rolling_median, linewidth=2.0, color="tab:green", label=f"Rolling median ({window})")
        ax.fill_between(rolling_x, q25, q75, color="tab:green", alpha=0.2, label=f"IQR 25/75 ({window})")

    _draw_split_lines(ax, splits, split_labels)

    ax.set_ylabel("Final Score")
    ax.grid(True, alpha=0.3)
    h,l = ax.get_legend_handles_labels()
    ax.legend(loc="upper left", fontsize=8, handles=h[:4], labels=l[:4])


def _draw_failure_rate(ax, rows, rolling_window, splits, split_labels, failure_threshold):
    x = list(range(1, len(rows) + 1))
    flags, fallback_count = _get_failure_flags(rows, failure_threshold)

    import numpy as np

    flags_arr = np.array(flags, dtype=float)
    cumulative_rate = np.cumsum(flags_arr) / np.arange(1, len(flags_arr) + 1)
    ax.plot(x, cumulative_rate, linewidth=2.0, color="tab:orange", label="Cumulative failure rate")

    if len(rows) >= 2:
        window = max(2, min(int(rolling_window), len(flags)))
        kernel = np.ones(window) / window
        rolling_rate = np.convolve(flags_arr, kernel, mode="valid")
        rolling_x = list(range(window, len(flags) + 1))
        ax.plot(rolling_x, rolling_rate, linewidth=1.8, color="tab:red", label=f"Rolling failure rate ({window})")

    _draw_split_lines(ax, splits, split_labels)

    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Failure rate")
    ax.grid(True, alpha=0.3)
    h,l = ax.get_legend_handles_labels()
    ax.legend(loc="upper left", fontsize=8, handles=h[:4], labels=l[:4])

    if fallback_count > 0:
        ax.text(
            0.01,
            0.02,
            f"Used score<threshold fallback for {fallback_count} run(s)",
            transform=ax.transAxes,
            fontsize=8,
            va="bottom",
            ha="left",
            color="dimgray",
        )


def export_failure_rate_plot(rows, output_png, title, rolling_window, splits, split_labels, failure_threshold):
    output_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    _draw_failure_rate(ax, rows, rolling_window, splits, split_labels, failure_threshold)
    ax.set_xlabel("Run index (time-ordered)")
    ax.set_title(title)

    h,l = ax.get_legend_handles_labels()
    ax.legend(loc="upper left", fontsize=8, handles=h[:2], labels=l[:2])

    fig.tight_layout()
    fig.savefig(output_png, dpi=150)
    plt.close(fig)


def export_additive_plot(
    rows,
    output_png,
    title,
    failure_title,
    rolling_window,
    splits,
    split_labels,
    failure_threshold,
):
    output_png.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_score, ax_failure) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(11, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    _draw_score_progress(ax_score, rows, rolling_window, splits, split_labels)
    _draw_failure_rate(ax_failure, rows, rolling_window, splits, split_labels, failure_threshold)

    ax_score.set_title(title)
    ax_failure.set_title(failure_title)
    ax_failure.set_xlabel("Run index (time-ordered)")

    fig.tight_layout()
    fig.savefig(output_png, dpi=150)
    plt.close(fig)


def export_progress_plot(rows, output_png, title, rolling_window, splits, split_labels=None):
    output_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    _draw_score_progress(ax, rows, rolling_window, splits, split_labels)
    ax.set_xlabel("Run index (time-ordered)")
    ax.set_title(title)

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
        "--failure-title",
        default="Run Failure Rate Over Time",
        help="Title for failure-rate plot",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=10,
        help="Rolling average window size for the trend plot",
    )
    parser.add_argument(
        "--output-failure-png",
        default=str(Path(__file__).resolve().parent / "run_failure_rate_over_time.png"),
        help="Output PNG path for failure-rate trend",
    )
    parser.add_argument(
        "--output-additive-png",
        default=str(Path(__file__).resolve().parent / "run_score_and_failure_over_time.png"),
        help="Output PNG path for combined score+failure additive chart",
    )
    parser.add_argument(
        "--failure-threshold",
        type=float,
        default=0.0,
        help=(
            "Score threshold used only when Run/success is unavailable. "
            "A run is considered failed when final_score < failure_threshold."
        ),
    )
    parser.add_argument(
        "--failure-plot-mode",
        choices=["separate", "additive", "both"],
        default="separate",
        help=(
            "How to render failure-rate visuals: "
            "'separate' writes a dedicated failure-rate image, "
            "'additive' writes one combined score+failure image to --output-progress-png, "
            "'both' writes both outputs."
        ),
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
        
            
    export_csv(rows, Path(args.output_csv), args.failure_threshold)
    export_plot(rows, Path(args.output_png), args.title)
    if args.failure_plot_mode in ("separate", "both"):
        export_progress_plot(
            rows,
            Path(args.output_progress_png),
            args.progress_title,
            args.rolling_window,
            splits,
            split_labels,
        )
        export_failure_rate_plot(
            rows,
            Path(args.output_failure_png),
            args.failure_title,
            args.rolling_window,
            splits,
            split_labels,
            args.failure_threshold,
        )

    if args.failure_plot_mode in ("additive", "both"):
        additive_output = Path(args.output_progress_png) if args.failure_plot_mode == "additive" else Path(args.output_additive_png)
        export_additive_plot(
            rows,
            additive_output,
            args.progress_title,
            args.failure_title,
            args.rolling_window,
            splits,
            split_labels,
            args.failure_threshold,
        )

    print(f"Exported {len(rows)} runs")
    print(f"CSV: {args.output_csv}")
    print(f"Plot: {args.output_png}")
    print(f"Trend plot: {args.output_progress_png}")
    if args.failure_plot_mode in ("separate", "both"):
        print(f"Failure-rate plot: {args.output_failure_png}")
    if args.failure_plot_mode in ("additive", "both"):
        additive_print_path = args.output_progress_png if args.failure_plot_mode == "additive" else args.output_additive_png
        print(f"Additive plot: {additive_print_path}")


if __name__ == "__main__":
    main()
