"""Prepare evidence-checked special-phrasing continuation data and holdout."""
import collections
from decimal import Decimal
import json
from pathlib import Path
import random
import re

import prepare_supervision as sup


ROOT = sup.ROOT
V1 = ROOT / "runs/qwen2b_parse_once_sft_v1"
OUT = ROOT / "runs/qwen2b_parse_once_sft_v2"
SEED = 20261022
SPECIAL_TRAIN = 256
REPLAY = 256
HOLDOUT = 120
PATTERNS = {
    "sideways": r"\bsideways\b|side-facing",
    "distance_away": r"\b\d+(?:\.\d+)?\s+(?:meters|metres)\s+away\b",
    "at_distance": r"\bat a distance of\b",
    "relative_clause": r"\b(?:that|these|those|whose)\b",
    "between": r"\bbetween\b",
    "stand": r"\bstand\b",
    "about": r"\babout\b",
    "object_name": r"\bobject name\b",
    "extra_filler": r"\b(?:clearly|visible|observed|characterized|structure|road|traffic|designated)\b",
}


def numbers(text):
    return [Decimal(x) for x in re.findall(r"\d+(?:\.\d+)?", text)]


def evidence_target(annotation):
    text = annotation["text"].lower().replace("centre", "center")
    props = annotation["properties"]
    nums = numbers(text)
    result = {}
    for key, value in props.items():
        if key in ("length", "width", "height"):
            lo, hi = map(Decimal, re.findall(r"\d+(?:\.\d+)?", value))
            if lo not in nums or hi not in nums: raise ValueError(f"numeric_evidence:{key}")
            result[f"{key}_min_m"] = float(lo); result[f"{key}_max_m"] = float(hi)
        elif key == "distance":
            distance = Decimal(value.split()[0])
            if distance not in nums: raise ValueError("numeric_evidence:distance")
            result["distance_max_m"] = float(distance)
        elif key == "objectName":
            pattern = r"\b" + re.escape(value) + (r"(?:s|es)?\b" if value != "bus" else r"(?:es)?\b")
            if not re.search(pattern, text): raise ValueError("text_evidence:category")
            result["category"] = value
        elif key == "appearance":
            if value not in text and value.replace("-", " ") not in text: raise ValueError("text_evidence:color")
            result["color"] = value
        elif key == "place":
            if not re.search(r"\b" + re.escape(value) + r"\b", text): raise ValueError("text_evidence:position")
            result["position"] = value
        elif key == "facing_direction":
            if value == "side facing me": ok = bool(re.search(r"side[- ]facing me|sideways towards me", text))
            elif value == "facing away from me": ok = "facing away from me" in text
            else: ok = bool(re.search(r"\b(?:facing|face) (?:towards )?me\b", text)) and "side" not in text
            if not ok: raise ValueError("text_evidence:facing")
            result["facing"] = value
        elif key == "occlusion":
            if value == "partially occluded": ok = "partially occluded" in text
            else: ok = bool(re.search(r"not occluded|no occlusion|without (?:any )?(?:being )?(?:occluded|occlusion)", text))
            if not ok: raise ValueError("text_evidence:occlusion")
            result["occlusion"] = value
        else:
            raise ValueError(f"unknown_property:{key}")
    if not result: raise ValueError("empty_target")
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    split = sup.read(ROOT / "runs/20261015_qwen2b_input_v1/split_manifest.json")
    train_files = set(split["train"])
    records = sup.read(ROOT / "runs/20261015_qwen2b_data_audit/training_records.json")
    candidates = []; rejected = collections.Counter(); pattern_counts = collections.Counter()
    for record in records:
        if record["file"] not in train_files: continue
        for index, annotation in enumerate(record["annotations"]):
            kinds = [name for name, pattern in PATTERNS.items() if re.search(pattern, annotation["text"], re.I)]
            if not kinds: continue
            try:
                target = evidence_target(annotation)
                row = {"source_file": record["file"], "camera": record["camera"], "annotation_index": index,
                       "description": annotation["text"], "target": target, "special_kinds": kinds}
                candidates.append(row); pattern_counts.update(kinds)
            except ValueError as error:
                rejected[str(error)] += 1
    by_file = collections.defaultdict(list)
    for row in candidates: by_file[row["source_file"]].append(row)
    rng = random.Random(SEED); files = sorted(by_file); rng.shuffle(files)
    holdout_files = set(files[:HOLDOUT])
    holdout = [rng.choice(by_file[name]) for name in files[:HOLDOUT]]
    remaining = files[HOLDOUT:]; special_train = [rng.choice(by_file[name]) for name in remaining[:SPECIAL_TRAIN]]
    assert len(holdout) == HOLDOUT and len(special_train) == SPECIAL_TRAIN
    replay_pool = [x for x in sup.read(V1 / "train.json") if x["source_file"] not in holdout_files]
    rng.shuffle(replay_pool); replay = replay_pool[:REPLAY]
    assert len(replay) == REPLAY
    training = []
    for row in special_train:
        training.append({**row, "source": "special"})
    for row in replay:
        training.append({"source_file": row["source_file"], "camera": row["camera"],
                         "annotation_index": row["annotation_index"], "description": row["description"],
                         "target": row["target"], "special_kinds": [], "source": "replay"})
    rng.shuffle(training)
    for i, row in enumerate(training, 1): row["sample_id"] = f"v2-train-{i:04d}"
    for i, row in enumerate(holdout, 1): row["sample_id"] = f"v2-holdout-{i:03d}"
    assert not {x["source_file"] for x in training} & {x["source_file"] for x in holdout}
    sup.write(OUT / "train.json", training); sup.write(OUT / "special_holdout.json", holdout)
    manifest = {"status": "PREPARED", "seed": SEED, "evidence_checked_special_candidates": len(candidates),
                "candidate_files": len(by_file), "special_train": SPECIAL_TRAIN, "replay": REPLAY,
                "training_samples": len(training), "special_holdout": len(holdout), "train_holdout_file_overlap": 0,
                "candidate_pattern_counts": dict(pattern_counts),
                "selected_train_pattern_counts": dict(collections.Counter(k for x in special_train for k in x["special_kinds"])),
                "holdout_pattern_counts": dict(collections.Counter(k for x in holdout for k in x["special_kinds"])),
                "rejected": dict(rejected), "train_sha256": sup.digest(OUT / "train.json"),
                "holdout_sha256": sup.digest(OUT / "special_holdout.json"),
                "parent_adapter_sha256": {p.name: sup.digest(p) for p in (V1 / "adapter").iterdir() if p.is_file()},
                "policy": "Every target field requires direct textual evidence; numeric values must exactly match training annotation values."}
    sup.write(OUT / "manifest.json", manifest)
    sup.write(OUT / "status.json", {"status": "PREPARED", "completed_steps": 0, "total_steps": len(training)})
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
