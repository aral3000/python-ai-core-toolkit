#!/usr/bin/env python3
"""
Challenge Runner & Progress Tracker
Usage: python run_all_challenges.py
"""

import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent

CHALLENGES = [
    ("c01_data_structures", "Challenge 01: O(1) LRU Cache (Data Structures)"),
    ("c02_numpy_vectorization", "Challenge 02: NumPy AI Vectorization Kernels"),
    ("c03_pandas_pipeline", "Challenge 03: Production Pandas Feature Engineering")
]


def run():
    print("\n" + "=" * 65)
    print(" PYTHON AI CORE TOOLKIT - TDD PROGRESS TRACKER")
    print("=" * 65)
    
    results = []
    for pkg, name in CHALLENGES:
        test_file = ROOT / "challenges" / pkg
        cmd = [sys.executable, "-m", "pytest", str(test_file), "-q"]
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        passed = (proc.returncode == 0)
        results.append((name, passed, proc.stdout.strip()))

    print("\nCURRENT STATUS:")
    for name, passed, output in results:
        status_badge = "[PASSED]" if passed else "[PENDING / FAILING]"
        print(f" {status_badge:<20} {name}")

    print("\n" + "-" * 65)
    print("Instructions for Athallah:")
    print("1. Open the file in challenges/c0X_.../ and implement the TODO sections.")
    print("2. Run individual tests: pytest challenges/c0X_.../test_*.py")
    print("3. When all 3 pass, commit and push your solution to GitHub!")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    run()
