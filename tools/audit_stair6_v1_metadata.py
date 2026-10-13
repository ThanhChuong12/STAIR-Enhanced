"""Read-only metadata preflight; never guesses ASIN mappings or categories.

Missing inputs produce a machine-readable NO-GO result without importing the
training runtime. Full validation uses the same category loader as the model.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_dataset(dataset_dir, mapping_file=None, metadata_file=None, manifest_file=None,
                  n_items=None, dataset=None, feature_files=None):
    directory = Path(dataset_dir)
    if not directory.is_dir():
        return {"ready": False, "status": "NO-GO", "error": f"Dataset directory not found: {directory}"}
    paths = {
        "mapping": Path(mapping_file) if mapping_file else directory / "item_id_to_asin.json",
        "metadata": Path(metadata_file) if metadata_file else directory / "item_categories.jsonl",
        "manifest": Path(manifest_file) if manifest_file else directory / "category_metadata_manifest.json",
    }
    result = {
        "schema_version": 1, "dataset": dataset or directory.name,
        "dataset_dir": str(directory.resolve()), "ready": False, "status": "NO-GO",
        "inventory": sorted(p.name for p in directory.iterdir() if p.is_file()),
        "required_inputs": {name: str(path) for name, path in paths.items()},
        "missing_inputs": [name for name, path in paths.items() if not path.is_file()],
        "scope": "Structural/provenance checks; provider mapping attestation is not a semantic alignment proof.",
    }
    if result["missing_inputs"]:
        return result
    feature_files = feature_files or ("textual_modality.pkl", "visual_modality.pkl")
    missing_data = [name for name in (*feature_files, "train.txt", "valid.txt", "test.txt")
                    if not (directory / name).is_file()]
    if missing_data:
        result["error"] = f"Missing baseline data files: {missing_data}"
        return result
    try:
        item_ids = set()
        split_rows = {}
        for name in ("train.txt", "valid.txt", "test.txt"):
            count = 0
            with (directory / name).open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream, delimiter="\t")
                if "ITEM" not in (reader.fieldnames or ()):
                    raise ValueError(f"{name} must use the native ITEM column")
                for row in reader:
                    if not row["ITEM"].isdecimal():
                        raise ValueError(f"{name} ITEM IDs must be processed nonnegative integers")
                    item_ids.add(int(row["ITEM"]))
                    count += 1
            split_rows[name] = count
        if not item_ids:
            raise ValueError("Baseline splits contain no items")
        catalog_size = max(item_ids) + 1 if n_items is None else n_items
        if catalog_size <= max(item_ids):
            raise ValueError("Declared item count cannot cover the processed split IDs")
        expected = {
            "dataset": result["dataset"],
            "features": {name: file_digest(directory / name) for name in feature_files},
            "splits": {p.name: file_digest(p) for p in sorted(directory.iterdir())
                       if p.is_file() and p.suffix in (".txt", ".csv")},
        }
        from models.stair6_v1_categories import load_category_data
        category_data = load_category_data(paths["mapping"], paths["metadata"], paths["manifest"],
                                           catalog_size, expected_provenance=expected)
        informative = category_data.audit["informative_items"] > 0
        result.update(n_items=catalog_size, split_rows=split_rows, audit=category_data.audit,
                      metadata_fingerprint=category_data.metadata_fingerprint,
                      baseline=expected, ready=informative,
                      status="READY_FOR_GRAPH_DIAGNOSTICS" if informative else "NO-GO")
        if not informative:
            result["error"] = "Metadata has no informative terminal category paths"
    except (ValueError, OSError, KeyError, TypeError) as error:
        result["error"] = str(error)
    return result


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--dataset")
    parser.add_argument("--mapping-file")
    parser.add_argument("--metadata-file")
    parser.add_argument("--manifest-file")
    parser.add_argument("--n-items", type=int)
    parser.add_argument("--feature-files", default="textual_modality.pkl,visual_modality.pkl")
    parser.add_argument("--output", help="Optional new JSON audit artifact")
    args = parser.parse_args(arguments)
    result = audit_dataset(args.dataset_dir, args.mapping_file, args.metadata_file, args.manifest_file,
                           args.n_items, args.dataset, args.feature_files.split(","))
    serialized = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        output = Path(args.output)
        inputs = {Path(path).resolve() for path in result.get("required_inputs", {}).values()}
        directory = Path(args.dataset_dir)
        if directory.is_dir():
            inputs.update(path.resolve() for path in directory.iterdir() if path.is_file())
        if output.resolve() in inputs:
            parser.error("Audit output must not overwrite an input or baseline dataset file")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0 if result["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
