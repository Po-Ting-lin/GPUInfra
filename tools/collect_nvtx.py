#!/usr/bin/env python3
"""Collect matching batched/interleaved demo traces without CPU sampling."""
import argparse
import pathlib
import shutil
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--binary", type=pathlib.Path, default=pathlib.Path("cmake-build-nvtx-on/gpuinfra_demo"))
parser.add_argument("--output", type=pathlib.Path, required=True)
parser.add_argument("--timed-frames", type=int, default=200)
parser.add_argument("--warmup-frames", type=int, default=20)
parser.add_argument("--size-factor", type=int, default=64)
args = parser.parse_args()
binary = args.binary.resolve()
nsys = shutil.which("nsys")
if nsys is None or not binary.is_file():
    parser.error("nsys and an NVTX-enabled demo binary are required")
args.output.mkdir(parents=True, exist_ok=False)
output = args.output.resolve()
reports = "nvtx_sum,nvtx_pushpop_trace,cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum"
with (output / "commands.txt").open("w") as commands:
    for mode in ("batched", "interleaved"):
        command = [nsys, "profile", "--trace=cuda,nvtx", "--sample=none", "--cpuctxsw=none", "--force-overwrite=false", "--output=" + str(output / mode), str(binary), str(args.timed_frames), str(args.warmup_frames), mode, str(args.size_factor)]
        commands.write(repr(command) + "\n")
        commands.flush()
        with (output / (mode + ".log")).open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        command = [nsys, "stats", "--report=" + reports, "--format=csv", "--output=" + str(output / mode), str(output / (mode + ".nsys-rep"))]
        commands.write(repr(command) + "\n")
        commands.flush()
        with (output / (mode + ".stats.log")).open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
with (output / "environment.txt").open("w") as log:
    subprocess.run([nsys, "--version"], stdout=log, check=True)
    subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv"], stdout=log, check=True)
