"""Prepare a deterministic description-to-flat-JSON LoRA pilot dataset."""
import collections
import json
from pathlib import Path
import random

import prepare_ab
import prepare_supervision as sup
import run_parse_once as base


ROOT = sup.ROOT
OUT = ROOT / "runs/qwen2b_parse_once_sft_v1"
SEED = 20261021
PILOT_SIZE = 512


def flat(parsed):
    result = {}
    mapping = {
        "objectName": "category", "appearance": "color", "place": "position",
        "facing_direction": "facing", "occlusion": "occlusion",
    }
    for key, value in parsed.items():
        if key in mapping:
            result[mapping[key]] = value
        elif key == "distance":
            result["distance_max_m"] = value
        elif key in ("length", "width", "height"):
            result[f"{key}_min_m"], result[f"{key}_max_m"] = value
        else:
            raise ValueError(key)
    assert result and set(result) <= base.ALLOWED_KEYS
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    records = sup.read(ROOT / "runs/20261015_qwen2b_data_audit/training_records.json")
    split = sup.read(ROOT / "runs/20261015_qwen2b_input_v1/split_manifest.json")
    train_files, validation_files = set(split["train"]), set(split["validation"])
    assert not train_files & validation_files
    formal_validation = sup.read(ROOT / "runs/qwen2b_supervision_ab_expanded_v1/validation.json")
    formal_files = {x["source_file"] for x in formal_validation}
    assert formal_files <= validation_files and not formal_files & train_files

    eligible, rejected = [], collections.Counter()
    for record in records:
        if record["file"] not in train_files:
            continue
        for annotation_index, annotation in enumerate(record["annotations"]):
            try:
                parsed = prepare_ab.extract(annotation["text"])
                if parsed != prepare_ab.normalized(annotation["properties"]):
                    raise ValueError("text_property_disagreement")
                target = flat(parsed)
                eligible.append({
                    "source_file": record["file"], "camera": record["camera"],
                    "annotation_index": annotation_index, "description": annotation["text"],
                    "target": target, "signature": sorted(target),
                })
            except (ValueError, AssertionError) as error:
                rejected[str(error) or "unknown"] += 1
    assert len(eligible) == 7231

    rng = random.Random(SEED)
    groups = collections.defaultdict(list)
    for row in eligible:
        # Generic/specific category is kept in the signature because it is the
        # dominant observed zero-shot failure.
        groups[(tuple(row["signature"]), "specific" if "category" in row["target"] else "generic")].append(row)
    for rows in groups.values():
        rng.shuffle(rows)
    keys = sorted(groups, key=str)
    rng.shuffle(keys)
    selected, used = [], set()
    while len(selected) < PILOT_SIZE:
        progressed = False
        for key in keys:
            while groups[key] and (groups[key][-1]["source_file"], groups[key][-1]["annotation_index"]) in used:
                groups[key].pop()
            if not groups[key]:
                continue
            row = groups[key].pop()
            identity = (row["source_file"], row["annotation_index"])
            used.add(identity); selected.append(row); progressed = True
            if len(selected) == PILOT_SIZE:
                break
        if not progressed:
            raise RuntimeError("Not enough eligible rows")
    rng.shuffle(selected)
    for index, row in enumerate(selected, 1):
        row["sample_id"] = f"sft-{index:04d}"

    sup.write(OUT / "train.json", selected)
    fields = collections.Counter(key for row in selected for key in row["target"])
    generic = sum("category" not in row["target"] for row in selected)
    manifest = {
        "status": "PREPARED", "seed": SEED, "pilot_samples": len(selected),
        "eligible_train_descriptions": len(eligible), "source_train_files": len(train_files),
        "formal_validation_files": len(formal_files), "train_formal_file_overlap": 0,
        "generic_vehicle_samples": generic, "specific_category_samples": len(selected) - generic,
        "field_coverage": dict(sorted(fields.items())), "condition_signatures": len({tuple(x["signature"]) for x in selected}),
        "train_sha256": sup.digest(OUT / "train.json"),
        "source_records_sha256": sup.digest(ROOT / "runs/20261015_qwen2b_data_audit/training_records.json"),
        "split_sha256": sup.digest(ROOT / "runs/20261015_qwen2b_input_v1/split_manifest.json"),
        "formal_validation_sha256": sup.digest(ROOT / "runs/qwen2b_supervision_ab_expanded_v1/validation.json"),
        "system_prompt_sha256": __import__("hashlib").sha256(base.SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "screening": "Only conservative parser outputs exactly matching annotation properties are eligible.",
        "rejected": dict(rejected),
    }
    sup.write(OUT / "manifest.json", manifest)
    sup.write(OUT / "status.json", {"status": "PREPARED", "completed_steps": 0, "total_steps": PILOT_SIZE})
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
