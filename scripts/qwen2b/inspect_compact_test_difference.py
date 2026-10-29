"""Locate candidate-level differences between compact and frozen V2 test outputs."""
import json
from pathlib import Path

import predict_test_a as test_a
import predict_test_parse_once_v2 as old_test
import prepare_supervision as sup
import run_parse_once as base


ROOT = sup.ROOT
NEW_DIR = ROOT / "runs/qwen2b_test_compact_v2"
OLD_DIR = ROOT / "runs/qwen2b_test_parse_once_v2"


def main():
    predictions = sup.read(NEW_DIR / "predictions.json")
    new_by_text = {row["description"]: row for row in predictions}

    old_rows = [
        json.loads(line)
        for line in (OLD_DIR / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    resolution = sup.read(OLD_DIR / "resolution.json")
    for row in old_rows:
        if row["description_id"] == resolution["description_id"]:
            row["parsed"] = resolution["resolved_conditions"]
    old_by_text = {row["description"]: row for row in old_rows}

    manifest = sup.read(OLD_DIR / "manifest.json")
    differences = []
    for entry in manifest["files"]:
        path = ROOT / "data/raw/MonoMulti3D/test" / entry["file"]
        data = sup.read(path)
        candidates = test_a.objects(data, entry["width"], entry["height"])
        for candidate_index, candidate in enumerate(candidates):
            for description_index, text in enumerate(data["public_description"]):
                new_conditions = new_by_text[text]["resolved"]
                old_conditions = old_by_text[text]["parsed"]
                new_value, new_reasons = base.match_candidate(new_conditions, candidate)
                old_value, old_reasons = base.match_candidate(old_conditions, candidate)
                if int(new_value) != int(old_value):
                    differences.append({
                        "file": entry["file"],
                        "description_index": description_index,
                        "description": text,
                        "candidate_index": candidate_index,
                        "candidate": candidate,
                        "new_conditions": new_conditions,
                        "old_conditions": old_conditions,
                        "new_value": int(new_value),
                        "old_value": int(old_value),
                        "new_reasons": new_reasons,
                        "old_reasons": old_reasons,
                    })

    sup.write(NEW_DIR / "decision_differences.json", differences)
    print(json.dumps(differences, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
