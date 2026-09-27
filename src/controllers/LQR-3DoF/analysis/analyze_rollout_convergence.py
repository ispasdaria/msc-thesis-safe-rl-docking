"""Analyze cumulative convergence of saved LQR mission rollout records.

Example:
    python analyze_rollout_convergence.py \
        --input results/lqr_mission/legacy_full_mission_validation_dtclose_1p0_dtfinal_1p0/seeds_300_599/validation_300_599.csv \
        --output results/lqr_mission/legacy_full_mission_validation_dtclose_1p0_dtfinal_1p0/seeds_300_599/validation_300_599_convergence.csv \
        --metric closing_final_error_norm

The input is the per-mission CSV written by run_lqr_full_mission.py. The
analysis uses cumulative prefixes and repeated random permutations, so it does
not depend on one arbitrary seed ordering.
"""

import argparse
import csv
import json
import math
import random
from pathlib import Path


DEFAULT_SIZES = (10, 20, 30, 40, 50, 75, 100, 150, 200, 250, 300)


def mean_and_variance(values):
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, float("nan")
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, variance


def wilson_interval(successes, total, z=1.96):
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    half_width = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return centre - half_width, centre + half_width, half_width


def read_records(input_path):
    with open(input_path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def metric_values(records, metric):
    values = []
    for record in records:
        value = record.get(metric, "")
        if value in ("", "None", "nan", "NaN"):
            continue
        values.append(float(value))
    return values


def summarize_prefix(records, metric, sample_size):
    prefix = records[:sample_size]
    values = metric_values(prefix, metric)
    mean, variance = mean_and_variance(values) if values else (float("nan"), float("nan"))
    success_key = "mission_success" if "mission_success" in prefix[0] else "success"
    successes = sum(record[success_key].lower() == "true" for record in prefix)
    if "closing_reason" in prefix[0]:
        safety_violations = sum(
            record["closing_reason"] == "safety_violation"
            or record["final_reason"] == "safety_violation"
            for record in prefix
        )
    else:
        safety_violations = sum(
            record.get("termination_reason") == "safety_violation"
            for record in prefix
        )
    lower, upper, half_width = wilson_interval(successes, len(prefix))
    return {
        "n": sample_size,
        "success_rate": successes / sample_size,
        "success_wilson_lower": lower,
        "success_wilson_upper": upper,
        "success_wilson_half_width": half_width,
        "safety_rate": safety_violations / sample_size,
        "metric_mean": mean,
        "metric_sample_variance": variance,
    }


def analyze(records, metric, sample_sizes, permutations):
    usable_sizes = [size for size in sample_sizes if size <= len(records)]
    rows = []
    for sample_size in usable_sizes:
        observed = summarize_prefix(records, metric, sample_size)
        permutation_rows = []
        for permutation_index in range(permutations):
            shuffled = list(records)
            random.Random(1000 + permutation_index).shuffle(shuffled)
            permutation_rows.append(summarize_prefix(shuffled, metric, sample_size))

        for key in (
            "success_rate",
            "safety_rate",
            "metric_mean",
            "metric_sample_variance",
            "success_wilson_half_width",
        ):
            values = [row[key] for row in permutation_rows if math.isfinite(row[key])]
            if values:
                values.sort()
                observed[f"{key}_permutation_median"] = values[len(values) // 2]
                observed[f"{key}_permutation_p05"] = values[max(0, int(0.05 * (len(values) - 1)))]
                observed[f"{key}_permutation_p95"] = values[int(0.95 * (len(values) - 1))]
        rows.append(observed)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metric", default="closing_final_error_norm")
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--sample-sizes", nargs="+", type=int, default=DEFAULT_SIZES)
    args = parser.parse_args()

    records = read_records(args.input)
    if not records:
        raise ValueError("The input CSV contains no mission records.")

    rows = analyze(records, args.metric, args.sample_sizes, args.permutations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    metadata_path = args.output.with_suffix(".json")
    metadata = {
        "input": str(args.input),
        "metric": args.metric,
        "num_records": len(records),
        "permutations": args.permutations,
        "sample_sizes": [row["n"] for row in rows],
        "output": str(args.output),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")
    print(f"wrote {metadata_path}")


if __name__ == "__main__":
    main()
