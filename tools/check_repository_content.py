#!/usr/bin/env python3
"""Reject raw profiler artifacts and recognizable credentials without printing values."""
import argparse
import re
import subprocess
import sys

RAW_SUFFIXES = (".sqlite", ".sqlite3", ".db", ".nsys-rep", ".qdrep", ".qdstrm", ".sqlite-wal", ".sqlite-shm", ".sqlite-journal")
SECRET_PATTERNS = (
    ("Hugging Face token", re.compile(rb"hf_[A-Za-z0-9]{20,}")),
    ("GitHub token", re.compile(rb"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})")),
    ("private key", re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA |ENCRYPTED )?PRIVATE KEY-----")),
)


def git(*args):
    return subprocess.check_output(["git", *args])


def check_entries(entries, label, checked):
    valid = True
    for path, object_id in entries:
        if path.lower().endswith(RAW_SUFFIXES):
            print(f"Blocked {label}: raw artifact {path!r}; keep it outside Git.", file=sys.stderr)
            valid = False
        if object_id in checked:
            continue
        checked.add(object_id)
        content = git("cat-file", "blob", object_id)
        for name, pattern in SECRET_PATTERNS:
            if pattern.search(content):
                print(f"Blocked {label}: {name} in {path!r} (value withheld).", file=sys.stderr)
                valid = False
    return valid


def staged_entries():
    for record in git("ls-files", "--stage", "-z").split(b"\0"):
        if record:
            metadata, path = record.split(b"\t", 1)
            mode, object_id, stage = metadata.split()
            if mode != b"160000":
                yield path.decode(errors="replace"), object_id.decode()


def commit_entries(commit):
    for record in git("ls-tree", "-r", "-z", commit).split(b"\0"):
        if record:
            metadata, path = record.split(b"\t", 1)
            mode, kind, object_id = metadata.split()
            if kind == b"blob":
                yield path.decode(errors="replace"), object_id.decode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-push", action="store_true", help="Read Git pre-push ref updates from stdin and scan outgoing history")
    args = parser.parse_args()
    checked = set()
    valid = True
    if not args.pre_push:
        valid = check_entries(staged_entries(), "index", checked)
    else:
        commits = set()
        for line in sys.stdin:
            local_ref, local_id, remote_ref, remote_id = line.split()
            if set(local_id) == {"0"}:
                continue
            revisions = [local_id]
            if set(remote_id) != {"0"}:
                revisions.append("^" + remote_id)
            commits.update(git("rev-list", *revisions).decode().splitlines())
        for commit in sorted(commits):
            valid = check_entries(commit_entries(commit), commit[:12], checked) and valid
    return 0 if valid else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except subprocess.CalledProcessError:
        print("Repository content check failed; push/commit blocked.", file=sys.stderr)
        sys.exit(1)
