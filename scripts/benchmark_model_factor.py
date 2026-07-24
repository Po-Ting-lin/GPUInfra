#!/usr/bin/env python3

import argparse
import csv
import random
import re
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MODELS = ("batched", "interleaved")
DEFAULT_FACTORS = tuple(range(16, 257, 16))
COLORS = {
    "batched": "#1f77b4",
    "interleaved": "#d62728",
}
TIMING_PATTERN = re.compile(
    r"timed (?P<timed>\d+) frames/GPU "
    r"\((?P<total>\d+) total\) in (?P<elapsed>[0-9.]+) ms"
)
THROUGHPUT_PATTERN = re.compile(
    r"throughput=(?P<throughput>[0-9.]+) frames/s, "
    r"average=(?P<average>[0-9.]+) ms/frame"
)
SIZING_PATTERN = re.compile(
    r"size_factor=(?P<factor>\d+) "
    r"input=(?P<input>\d+x\d+) "
    r"cel=(?P<cel>\d+x\d+) "
    r"sdd=(?P<sdd>\d+x\d+) "
    r"mi=(?P<mi>\d+x\d+)"
)
MODEL_PATTERN = re.compile(r"execution_model=(?P<model>\w+)")
GPU_PATTERN = re.compile(r"discovered gpu=\d+ name=(?P<name>.+?) numa=")
RAW_FIELDS = (
    "run",
    "repetition",
    "factor",
    "model",
    "elapsed_ms",
    "throughput_frames_per_second",
    "average_ms_per_frame",
    "timed_frames_per_gpu",
    "total_frames",
    "warmup_frames_per_gpu",
    "input_size",
    "cel_size",
    "sdd_size",
    "mi_size",
    "gpu_names",
)
SUMMARY_FIELDS = (
    "factor",
    "model",
    "samples",
    "median_elapsed_ms",
    "q1_elapsed_ms",
    "q3_elapsed_ms",
    "min_elapsed_ms",
    "max_elapsed_ms",
)


def parseArguments():
    repositoryRoot = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Benchmark batched and interleaved execution across image size factors."
    )
    parser.add_argument(
        "--executable",
        type=Path,
        default=repositoryRoot / "build" / "gpuinfra_demo",
        help="Path to gpuinfra_demo.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repositoryRoot / "benchmark_results",
        help="Parent directory for timestamped CSV and PNG artifacts.",
    )
    parser.add_argument(
        "--factors",
        type=int,
        nargs="+",
        default=list(DEFAULT_FACTORS),
        help="Factors to test; each must be a multiple of 16 from 16 through 256.",
    )
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--timed-frames", type=int, default=200)
    parser.add_argument("--warmup-frames", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save figures without calling matplotlib.pyplot.show().",
    )
    return parser.parse_args()


def validateArguments(args):
    if args.repetitions <= 0:
        raise ValueError("repetitions must be positive")
    if args.timed_frames <= 0:
        raise ValueError("timed frames must be positive")
    if args.warmup_frames < 0:
        raise ValueError("warmup frames cannot be negative")
    if not args.executable.is_file():
        raise ValueError(f"executable does not exist: {args.executable}")

    factors = sorted(set(args.factors))
    invalidFactors = [
        factor
        for factor in factors
        if factor < 16 or factor > 256 or factor % 16 != 0
    ]
    if invalidFactors:
        raise ValueError(f"invalid factors: {invalidFactors}")
    return factors


def requireMatch(pattern, output, description):
    match = pattern.search(output)
    if match is None:
        raise RuntimeError(f"could not parse {description} from benchmark output")
    return match


def runBenchmark(executable, repositoryRoot, timedFrames, warmupFrames, model, factor):
    command = [
        str(executable),
        str(timedFrames),
        str(warmupFrames),
        model,
        str(factor),
    ]
    completed = subprocess.run(
        command,
        cwd=repositoryRoot,
        capture_output=True,
        text=True,
        check=False,
    )
    output = completed.stderr + "\n" + completed.stdout
    if completed.returncode != 0:
        raise RuntimeError(
            f"benchmark failed with exit code {completed.returncode}\n{output}"
        )

    timing = requireMatch(TIMING_PATTERN, output, "timing")
    throughput = requireMatch(THROUGHPUT_PATTERN, output, "throughput")
    sizing = requireMatch(SIZING_PATTERN, output, "sizing")
    parsedModel = requireMatch(MODEL_PATTERN, output, "execution model").group("model")
    gpuNames = GPU_PATTERN.findall(output)

    parsedFactor = int(sizing.group("factor"))
    parsedTimedFrames = int(timing.group("timed"))
    totalFrames = int(timing.group("total"))
    if parsedModel != model or parsedFactor != factor:
        raise RuntimeError("benchmark output does not match the requested model and factor")
    if parsedTimedFrames != timedFrames or totalFrames != timedFrames:
        raise RuntimeError(
            f"expected {timedFrames} timed frames on one GPU, got "
            f"{parsedTimedFrames} frames/GPU and {totalFrames} total"
        )

    return {
        "factor": factor,
        "model": model,
        "elapsed_ms": float(timing.group("elapsed")),
        "throughput_frames_per_second": float(throughput.group("throughput")),
        "average_ms_per_frame": float(throughput.group("average")),
        "timed_frames_per_gpu": parsedTimedFrames,
        "total_frames": totalFrames,
        "warmup_frames_per_gpu": warmupFrames,
        "input_size": sizing.group("input"),
        "cel_size": sizing.group("cel"),
        "sdd_size": sizing.group("sdd"),
        "mi_size": sizing.group("mi"),
        "gpu_names": "; ".join(gpuNames),
    }


def summarize(rows, factors):
    samples = defaultdict(list)
    for row in rows:
        samples[(row["model"], row["factor"])].append(row["elapsed_ms"])

    summary = []
    for factor in factors:
        for model in MODELS:
            values = np.asarray(samples[(model, factor)], dtype=float)
            q1, median, q3 = np.percentile(values, [25, 50, 75])
            summary.append(
                {
                    "factor": factor,
                    "model": model,
                    "samples": len(values),
                    "median_elapsed_ms": median,
                    "q1_elapsed_ms": q1,
                    "q3_elapsed_ms": q3,
                    "min_elapsed_ms": np.min(values),
                    "max_elapsed_ms": np.max(values),
                }
            )
    return summary


def writeSummary(path, summary):
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary)


def plotResults(rows, summary, factors, repetitions, timedFrames, warmupFrames, scale):
    figure, axis = plt.subplots(figsize=(12, 7))
    for model in MODELS:
        modelRows = [row for row in rows if row["model"] == model]
        modelSummary = [row for row in summary if row["model"] == model]
        rawByFactor = defaultdict(list)
        for row in modelRows:
            rawByFactor[row["factor"]].append(row["elapsed_ms"])

        for factor in factors:
            values = rawByFactor[factor]
            offsets = np.linspace(-1.5, 1.5, len(values))
            axis.scatter(
                factor + offsets,
                values,
                color=COLORS[model],
                alpha=0.28,
                s=22,
                linewidths=0,
            )

        medians = np.asarray(
            [row["median_elapsed_ms"] for row in modelSummary], dtype=float
        )
        q1 = np.asarray([row["q1_elapsed_ms"] for row in modelSummary], dtype=float)
        q3 = np.asarray([row["q3_elapsed_ms"] for row in modelSummary], dtype=float)
        axis.errorbar(
            factors,
            medians,
            yerr=np.vstack((medians - q1, q3 - medians)),
            color=COLORS[model],
            marker="o",
            markersize=5,
            linewidth=2,
            capsize=3,
            label=f"{model.capitalize()} median",
        )

    gpuNames = rows[0]["gpu_names"] or "CUDA GPU"
    axis.set_title(
        f"{gpuNames}: execution model vs size factor\n"
        f"{timedFrames} timed frames, {warmupFrames} warmup, "
        f"median and IQR of {repetitions} runs"
    )
    axis.set_xlabel("Size factor F")
    axis.set_ylabel(f"Elapsed time for {timedFrames} total frames (ms)")
    axis.set_xticks(factors)
    axis.set_yscale(scale)
    if scale == "linear":
        axis.set_ylim(bottom=0)
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    return figure


def printSummary(summary, factors, timedFrames):
    indexed = {
        (row["model"], row["factor"]): row["median_elapsed_ms"]
        for row in summary
    }
    print(f"\nMedian elapsed time for {timedFrames} total frames:")
    print(" F    Batched (ms)    Interleaved (ms)    Delta I-B (ms)")
    for factor in factors:
        batched = indexed[("batched", factor)]
        interleaved = indexed[("interleaved", factor)]
        print(
            f"{factor:3d}    {batched:12.3f}    "
            f"{interleaved:16.3f}    {interleaved - batched:14.3f}"
        )


def main():
    args = parseArguments()
    factors = validateArguments(args)
    repositoryRoot = Path(__file__).resolve().parents[1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    runDirectory = args.output_dir / f"model_factor_{timestamp}"
    runDirectory.mkdir(parents=True)
    rawPath = runDirectory / "raw_timings.csv"
    summaryPath = runDirectory / "summary.csv"
    linearPath = runDirectory / "model_factor_linear.png"
    logPath = runDirectory / "model_factor_log.png"

    rows = []
    totalRuns = len(factors) * len(MODELS) * args.repetitions
    runNumber = 0
    with rawPath.open("w", newline="", encoding="utf-8") as rawOutput:
        writer = csv.DictWriter(rawOutput, fieldnames=RAW_FIELDS)
        writer.writeheader()

        for repetition in range(1, args.repetitions + 1):
            factorOrder = list(factors)
            random.Random(args.seed + repetition).shuffle(factorOrder)
            for factorIndex, factor in enumerate(factorOrder):
                if (repetition + factorIndex) % 2 == 0:
                    modelOrder = MODELS
                else:
                    modelOrder = tuple(reversed(MODELS))

                for model in modelOrder:
                    runNumber += 1
                    print(
                        f"[{runNumber:03d}/{totalRuns}] repetition={repetition} "
                        f"factor={factor} model={model}",
                        flush=True,
                    )
                    row = runBenchmark(
                        args.executable,
                        repositoryRoot,
                        args.timed_frames,
                        args.warmup_frames,
                        model,
                        factor,
                    )
                    row["run"] = runNumber
                    row["repetition"] = repetition
                    rows.append(row)
                    writer.writerow(row)
                    rawOutput.flush()
                    print(f"    elapsed={row['elapsed_ms']:.3f} ms", flush=True)

    summary = summarize(rows, factors)
    writeSummary(summaryPath, summary)
    linearFigure = plotResults(
        rows,
        summary,
        factors,
        args.repetitions,
        args.timed_frames,
        args.warmup_frames,
        "linear",
    )
    logFigure = plotResults(
        rows,
        summary,
        factors,
        args.repetitions,
        args.timed_frames,
        args.warmup_frames,
        "log",
    )
    linearFigure.savefig(linearPath, dpi=180)
    logFigure.savefig(logPath, dpi=180)
    printSummary(summary, factors, args.timed_frames)
    print(f"\nRaw CSV: {rawPath}")
    print(f"Summary CSV: {summaryPath}")
    print(f"Linear plot: {linearPath}")
    print(f"Log plot: {logPath}")

    if not args.no_show:
        plt.show()
    plt.close(linearFigure)
    plt.close(logFigure)


if __name__ == "__main__":
    main()
