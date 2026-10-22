"""Read-only coverage audit for official-test public descriptions."""
import collections
import json
from pathlib import Path

import prepare_ab


ROOT = Path(__file__).resolve().parents[2]
TEST = ROOT / "data/raw/MonoMulti3D/test"


def main():
    errors = collections.Counter()
    fields = collections.Counter()
    unsupported = []
    files = sorted(TEST.glob("*.json"))
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        for index, description in enumerate(data["public_description"]):
            try:
                parsed = prepare_ab.extract(description)
                fields.update(parsed.keys())
            except (ValueError, AssertionError) as error:
                errors[str(error)] += 1
                unsupported.append({"file": path.name, "index": index,
                                    "description": description, "error": str(error)})
    total = sum(len(json.loads(path.read_text(encoding="utf-8"))["public_description"]) for path in files)
    print(json.dumps({"test_files": len(files), "descriptions": total,
          "supported_by_conservative_parser": total - len(unsupported),
          "unsupported": len(unsupported), "field_mentions_in_supported": dict(fields),
          "error_counts": dict(errors), "unsupported_examples": unsupported[:10]},
          ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
