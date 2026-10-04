"""Aggregate complete paired corpus measurements without excluding unfavorable clips."""

import argparse
import json
from pathlib import Path

from npu_sr.benchmark import save_report
from npu_sr.quality_gate import paired_quality_summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--model", required=True)
    parser.add_argument("--json", required=True, type=Path)
    args = parser.parse_args()
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.reports]
    if not all(report["complete"] for report in reports):
        raise ValueError("Quality trials are incomplete")
    if len({report["corpus_definition_sha256"] for report in reports}) != 1:
        raise ValueError("Quality reports use different corpus definitions")
    declarations = {}
    for report in reports:
        for clip in report["preparation"]["clips"]:
            if clip["identifier"] in declarations and declarations[clip["identifier"]] != clip:
                raise ValueError("Repeated clip declaration differs between reports")
            declarations[clip["identifier"]] = clip
    clips = list(declarations.values())
    results = [row for report in reports for row in report["results"]]
    summary = paired_quality_summary(results, clips, args.model)
    save_report(summary, args.json)
    for group, row in summary["groups"].items():
        if group == "all" or group.startswith("split:"):
            print(group, row["clips"], "clips; gains:", row["gain"])
    print("Numeric quality gate:", summary["numeric_gate_passed"])
