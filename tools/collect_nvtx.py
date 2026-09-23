#!/usr/bin/env python3
"""Collect matching batched/interleaved demo traces without CPU sampling."""
import argparse
import os
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
# Raw captures may include system-wide process metadata even with a restricted
# child environment. Keep every new capture private to the current user.
os.umask(0o077)
args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
output = args.output.resolve()
# Reduce the child environment; this does not sanitize system metadata captured
# by Nsight. Raw reports and SQLite exports must never enter Git.
allowedEnvironment = ("PATH", "HOME", "USER", "LOGNAME", "TMPDIR", "LANG", "LC_ALL", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "CUDA_MODULE_LOADING")
profileEnvironment = {name: os.environ[name] for name in allowedEnvironment if name in os.environ}
reports = "nvtx_sum,nvtx_pushpop_trace,cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum"
with (output / "commands.txt").open("w") as commands:
    for mode in ("batched", "interleaved"):
        command = [nsys, "profile", "--trace=cuda,nvtx", "--sample=none", "--cpuctxsw=none", "--force-overwrite=false", "--output=" + str(output / mode), str(binary), str(args.timed_frames), str(args.warmup_frames), mode, str(args.size_factor)]
        commands.write(repr(command) + "\n")
        commands.flush()
        with (output / (mode + ".log")).open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, env=profileEnvironment)
        command = [nsys, "stats", "--report=" + reports, "--format=csv", "--output=" + str(output / mode), str(output / (mode + ".nsys-rep"))]
        commands.write(repr(command) + "\n")
        commands.flush()
        with (output / (mode + ".stats.log")).open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, env=profileEnvironment)
with (output / "environment.txt").open("w") as log:
    subprocess.run([nsys, "--version"], stdout=log, check=True, env=profileEnvironment)
    subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv"], stdout=log, check=True, env=profileEnvironment)
