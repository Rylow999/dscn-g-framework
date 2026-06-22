#!/usr/bin/env python3
"""
DSCN-G Verification Script
==========================
Run after simulation to verify all theorems against paper claims.
"""
import json, sys

def main():
    with open("data/base_simulation_100seeds.json") as f:
        data = json.load(f)
    s = data["summary"]

    checks = [
        ("Theorem 1: N* <= rho/theta_d", s["theorem1"]["verified"]),
        ("Theorem 2: |omega* - 0.649| < 0.10", s["theorem2"]["verified"]),
        ("Theorem 3: p_conv > 0.5", s["theorem3"]["verified"]),
        ("Prediction C3: hijack rate in [20, 35]%", 20 <= s["prediction3"]["hijack_rate_percent"] <= 35),
    ]

    all_pass = True
    print("=" * 65)
    print("DSCN-G v7.2 — Verification Report")
    print("=" * 65)
    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            all_pass = False

    print("=" * 65)
    if all_pass:
        print("VERIFICATION PASSED — Submission is internally consistent.")
        return 0
    else:
        print("VERIFICATION FAILED — Check outputs against paper claims.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
