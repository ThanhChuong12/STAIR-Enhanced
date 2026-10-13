"""Verified taxonomy inputs and train-only category association for STAIR6-v1.

CPU/offline computations query only category pairs needed on existing edges.
No dense item-item/category-category matrix or H h H.T product is constructed.
Hash binding verifies artifact identity, not semantic correctness of a provider's
ASIN mapping; obtaining that verified mapping remains a data prerequisite.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping
import unicodedata

import numpy as np
import scipy.sparse as sp

from models.stair5_v4_graph import binary_train_keys

CANONICALIZATION_VERSION = "terminal-full-path-nfc-v1"
_GENERIC = frozenset({"all", "all products", "products", "root", "unknown", "uncategorized", "none"})


@dataclass(frozen=True)
class CategoryData:
    memberships: sp.csr_matrix
    paths: tuple[tuple[str, ...], ...]
    item_asins: tuple[str, ...]
    audit: dict[str, Any]
    metadata_fingerprint: str
    provenance: dict[str, Any]


@dataclass(frozen=True)
class CategoryEvidence:
    pair_keys: np.ndarray
    counts: np.ndarray
    scores: np.ndarray
    metadata: dict[str, Any]


def _object_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON object key: {key!r}.")
        result[key] = value
    return result


def _read_json(text):
    def reject_constant(value):
        raise ValueError(f"Non-finite JSON constant: {value}.")
    return json.loads(text, object_pairs_hook=_object_pairs, parse_constant=reject_constant)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonicalize_paths(categories):
    """Keep distinct informative terminal paths and preserve parent context."""
    if not isinstance(categories, (list, tuple)):
        raise ValueError("categories must be a list of hierarchy paths.")
    paths = set()
    for path in categories:
        if not isinstance(path, (list, tuple)) or not path:
            raise ValueError("Every explicit category path must be a nonempty list.")
        labels = []
        for label in path:
            if not isinstance(label, str):
                raise ValueError("Category labels must be strings; brand/price are separate fields.")
            normalized = " ".join(unicodedata.normalize("NFC", label).split())
            if not normalized:
                raise ValueError("Category labels cannot be blank.")
            labels.append(normalized)
        terminal = tuple(labels)
        if len(terminal) > 1 and terminal[-1].casefold() not in _GENERIC:
            paths.add(terminal)
    return tuple(sorted(path for path in paths if not any(
        len(other) > len(path) and other[:len(path)] == path for other in paths)))


def build_memberships(category_paths, n_items=None):
    """Return (FP32 CSR [N_i,C], sorted full-path vocabulary).

    category_paths[i] contains the hierarchy paths of processed item i. A
    missing category is an empty list, not a guessed ASIN/category label.
    """
    if n_items is None:
        n_items = len(category_paths)
    if isinstance(n_items, bool) or not isinstance(n_items, (int, np.integer)) or n_items < 1:
        raise ValueError("n_items must be a positive integer.")
    if len(category_paths) != n_items:
        raise ValueError("Category rows must match the processed item catalog.")
    canonical = [canonicalize_paths(paths) for paths in category_paths]
    vocabulary = tuple(sorted({path for paths in canonical for path in paths}))
    ids = {path: index for index, path in enumerate(vocabulary)}
    rows, columns, values = [], [], []
    for item, paths in enumerate(canonical):
        if paths:
            rows.extend([item] * len(paths))
            columns.extend(ids[path] for path in paths)
            values.extend([1.0 / len(paths)] * len(paths))
    matrix = sp.csr_matrix((np.asarray(values, dtype=np.float32), (rows, columns)),
                          shape=(n_items, len(vocabulary)), dtype=np.float32)
    matrix.sort_indices()
    return matrix, vocabulary


def _require_text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string.")
    return value


def _digest_map(value, name):
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{name} must be a nonempty filename-to-SHA256 object.")
    for filename, digest in value.items():
        _require_text(filename, name)
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdefABCDEF" for c in digest):
            raise ValueError(f"Invalid SHA256 in {name}: {filename!r}.")
    return {key: digest.lower() for key, digest in value.items()}


def assert_category_provenance(category_data: CategoryData, expected_provenance):
    """Bind a subset of expected dataset/features/splits to the declared manifest.

    Expected features and splits may be supplied flat or inside 'baseline'.
    A supplied map must match the full corresponding manifest map exactly;
    omission permits staged dataset-first, actual-artifact-later validation.
    """
    if not isinstance(expected_provenance, Mapping):
        raise ValueError("expected_provenance must be a mapping.")
    manifest = category_data.provenance
    expected_baseline = expected_provenance.get("baseline", expected_provenance)
    if not isinstance(expected_baseline, Mapping):
        raise ValueError("Expected baseline provenance must be a mapping.")
    if "dataset" in expected_provenance and expected_provenance["dataset"] != manifest["dataset"]:
        raise ValueError("Category dataset provenance mismatch.")
    for name in ("features", "splits"):
        if name in expected_baseline:
            expected = _digest_map(expected_baseline[name], name)
            actual = _digest_map(manifest["baseline"][name], f"baseline.{name}")
            if expected != actual:
                raise ValueError(f"Category {name} provenance mismatch.")


def load_category_data(mapping_file, metadata_file, manifest_file, n_items,
                       expected_provenance=None):
    """Fail closed on catalog/provenance errors, neutral on missing item labels."""
    mapping_path, metadata_path, manifest_path = map(Path, (mapping_file, metadata_file, manifest_file))
    if isinstance(n_items, bool) or not isinstance(n_items, (int, np.integer)) or n_items < 1:
        raise ValueError("n_items must be a positive integer.")
    manifest = _read_json(manifest_path.read_text(encoding="utf-8-sig"))
    if not isinstance(manifest, dict) or type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise ValueError("Category manifest schema_version must be 1.")
    if type(manifest.get("n_items")) is not int or manifest["n_items"] != n_items:
        raise ValueError("Category manifest catalog size mismatch.")
    _require_text(manifest.get("dataset"), "manifest.dataset")
    if manifest.get("mapping_verified") is not True:
        raise ValueError("Category mapping must be explicitly verified by its provider.")
    _require_text(manifest.get("mapping_source"), "manifest.mapping_source")
    source = manifest.get("metadata_source")
    if isinstance(source, dict):
        _require_text(source.get("source"), "metadata_source.source")
        _require_text(source.get("snapshot"), "metadata_source.snapshot")
    else:
        _require_text(source, "manifest.metadata_source")
    files, baseline = manifest.get("files"), manifest.get("baseline")
    if not isinstance(files, dict) or not isinstance(baseline, dict):
        raise ValueError("Manifest requires files and baseline records.")
    _digest_map(baseline.get("features"), "baseline.features")
    _digest_map(baseline.get("splits"), "baseline.splits")
    mapping_digest, metadata_digest = _sha256(mapping_path), _sha256(metadata_path)
    if files.get("mapping_sha256") != mapping_digest or files.get("metadata_sha256") != metadata_digest:
        raise ValueError("Mapping/metadata SHA256 does not match the manifest.")
    mapping = _read_json(mapping_path.read_text(encoding="utf-8-sig"))
    if not isinstance(mapping, list) or len(mapping) != n_items:
        raise ValueError("Mapping must contain exactly one row per processed item.")
    if all(isinstance(row, str) for row in mapping):
        asins = [_require_text(row, "ASIN") for row in mapping]
    elif all(isinstance(row, dict) for row in mapping):
        asins = [None] * n_items
        for row in mapping:
            item = row.get("item_id")
            if type(item) is not int or not 0 <= item < n_items or asins[item] is not None:
                raise ValueError("Mapping item IDs must be unique contiguous catalog IDs.")
            asins[item] = _require_text(row.get("asin"), "ASIN")
    else:
        raise ValueError("Mapping must be an ASIN array or an item_id/asin record array.")
    if len(set(asins)) != n_items:
        raise ValueError("Mapping must be one-to-one; duplicate ASIN detected.")
    for asin in asins:
        if asin != asin.strip() or any(char.isspace() for char in asin):
            raise ValueError("ASINs cannot contain whitespace; never silently repair IDs.")
    category_rows = [[] for _ in range(n_items)]
    seen = set()
    with metadata_path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = _read_json(line)
            if not isinstance(record, dict):
                raise ValueError(f"Metadata line {line_number} must be an object.")
            item = record.get("item_id")
            if type(item) is not int or not 0 <= item < n_items or item in seen:
                raise ValueError(f"Invalid or duplicate metadata item_id at line {line_number}.")
            if record.get("asin") != asins[item]:
                raise ValueError(f"Metadata ASIN mismatch at line {line_number}.")
            if "categories" not in record:
                raise ValueError(f"Metadata line {line_number} requires categories (empty list is allowed).")
            category_rows[item] = canonicalize_paths(record["categories"])
            seen.add(item)
    memberships, paths = build_memberships(category_rows, n_items)
    per_item = np.diff(memberships.indptr)
    audit = {
        "n_items": int(n_items), "n_categories": len(paths), "metadata_rows": len(seen),
        "missing_metadata_rows": n_items - len(seen), "informative_items": int(np.count_nonzero(per_item)),
        "informative_coverage": float(np.count_nonzero(per_item) / n_items),
        "membership_nnz": int(memberships.nnz), "maximum_labels_per_item": int(per_item.max(initial=0)),
        "path_depth_histogram": {str(depth): sum(len(path) == depth for path in paths)
                                 for depth in sorted({len(path) for path in paths})},
        "canonicalization_version": CANONICALIZATION_VERSION,
        "mapping_verification_boundary": "Provider attestation and hash binding, not proof of semantic ASIN alignment.",
    }
    fingerprint = hashlib.sha256(json.dumps({"mapping": mapping_digest, "metadata": metadata_digest,
        "manifest": _sha256(manifest_path), "canonicalization": CANONICALIZATION_VERSION},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    result = CategoryData(memberships, paths, tuple(asins), audit, fingerprint, manifest)
    if expected_provenance is not None:
        assert_category_provenance(result, expected_provenance)
    return result


def _memberships(matrix):
    if not sp.issparse(matrix):
        raise ValueError("Memberships must be a sparse matrix.")
    matrix = matrix.tocsr(copy=True)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    if matrix.data.size and (not np.isfinite(matrix.data).all() or np.any(matrix.data <= 0)):
        raise ValueError("Memberships must be finite and nonnegative.")
    sums = np.asarray(matrix.sum(axis=1)).ravel()
    if not np.all((sums == 0) | np.isclose(sums, 1, atol=1e-6, rtol=0)):
        raise ValueError("Informative membership rows must sum to one.")
    return matrix


def _semantic(matrix, n_items):
    if not sp.issparse(matrix) or matrix.shape != (n_items, n_items):
        raise ValueError("Semantic graph must be a sparse square catalog matrix.")
    result = matrix.tocsr(copy=True)
    result.sum_duplicates()
    result.eliminate_zeros()
    result.sort_indices()
    if result.data.size and (not np.isfinite(result.data).all() or np.any(result.data < 0)):
        raise ValueError("Semantic graph must have finite nonnegative weights.")
    return result


def binary_user_category_incidence(train_edges, n_users, memberships):
    """Binary OR of categories across unique TRAIN user-item interactions."""
    membership = _memberships(memberships)
    n_items = membership.shape[0]
    keys = binary_train_keys(train_edges, n_users, n_items)
    interactions = sp.csr_matrix((np.ones(keys.size, dtype=np.int64),
        (keys // n_items, keys % n_items)), shape=(n_users, n_items), dtype=np.int64)
    binary_membership = membership.astype(np.int64)
    binary_membership.data[:] = 1
    incidence = (interactions @ binary_membership).tocsr()
    incidence.data[:] = 1
    incidence.eliminate_zeros()
    incidence.sort_indices()
    return incidence


def requested_category_pairs(raw_semantic, memberships, *, max_pairs=2_000_000,
                             max_combinations=20_000_000):
    """Bounded unordered category-pair queries needed on existing item edges."""
    membership = _memberships(memberships)
    semantic = _semantic(raw_semantic, membership.shape[0])
    n_categories = membership.shape[1]
    if max_pairs < 1 or max_combinations < 1 or n_categories ** 2 > np.iinfo(np.int64).max:
        raise ValueError("Invalid category pair workspace limit or int64 key overflow.")
    requested = set()
    combinations = 0
    symmetric = (semantic - semantic.T).nnz == 0
    for item in range(semantic.shape[0]):
        start, end = membership.indptr[item:item + 2]
        left = membership.indices[start:end]
        if not left.size:
            continue
        for neighbor in semantic.indices[semantic.indptr[item]:semantic.indptr[item + 1]]:
            if neighbor == item or (symmetric and neighbor < item):
                continue
            right = membership.indices[membership.indptr[neighbor]:membership.indptr[neighbor + 1]]
            combinations += left.size * right.size
            if combinations > max_combinations:
                raise MemoryError("Requested category combinations exceed the offline workspace limit.")
            for a in left:
                for b in right:
                    if a != b:
                        requested.add(int(min(a, b)) * n_categories + int(max(a, b)))
            if len(requested) > max_pairs:
                raise MemoryError("Requested category pairs exceed the offline workspace limit.")
    return np.asarray(sorted(requested), dtype=np.int64)


def category_association_scores(incidence, requested, min_count=2, shrinkage=5.0):
    """Bounded positive log-lift, counting only requested CSC posting pairs."""
    if not sp.issparse(incidence):
        raise ValueError("User-category incidence must be sparse.")
    if isinstance(min_count, bool) or not isinstance(min_count, (int, np.integer)) or min_count < 1:
        raise ValueError("min_count must be a positive integer.")
    if not math.isfinite(shrinkage) or shrinkage < 0:
        raise ValueError("shrinkage must be finite and nonnegative.")
    binary = incidence.tocsr(copy=True)
    binary.sum_duplicates()
    binary.eliminate_zeros()
    if binary.data.size and (not np.isfinite(binary.data).all() or np.any(binary.data != 1)):
        raise ValueError("Incidence must be binary, not weighted or event counts.")
    n_categories = binary.shape[1]
    keys = np.asarray(requested)
    if keys.ndim != 1 or keys.dtype.kind not in "iu":
        raise ValueError("Requested pair keys must be a one-dimensional integer array.")
    if keys.dtype.kind == "u" and keys.size and keys.max() > np.iinfo(np.int64).max:
        raise ValueError("Requested pair key exceeds int64.")
    keys = np.unique(keys.astype(np.int64, copy=False))
    if keys.size and (n_categories == 0 or keys.min() < 0 or keys.max() >= n_categories ** 2):
        raise ValueError("Requested category pair key is outside the category vocabulary.")
    if keys.size and np.any(keys // n_categories >= keys % n_categories):
        raise ValueError("Requested category pair keys must encode a < b (diagonal excluded).")
    population = int(np.count_nonzero(np.diff(binary.indptr)))
    postings = binary.tocsc()
    postings.sort_indices()
    frequencies = np.diff(postings.indptr).astype(np.int64)
    counts = np.zeros(keys.size, dtype=np.int64)
    scores = np.zeros(keys.size, dtype=np.float32)
    for index, key in enumerate(keys):
        a, b = divmod(int(key), n_categories)
        users_a = postings.indices[postings.indptr[a]:postings.indptr[a + 1]]
        users_b = postings.indices[postings.indptr[b]:postings.indptr[b + 1]]
        # CSC postings are sorted and unique. Search the smaller posting list
        # into the larger one instead of concatenating/sorting both for every
        # request; workspace is bounded by the smaller category population.
        smaller, larger = (users_a, users_b) if users_a.size <= users_b.size else (users_b, users_a)
        locations = np.searchsorted(larger, smaller)
        valid = locations < larger.size
        count = int(np.count_nonzero(larger[locations[valid]] == smaller[valid]))
        counts[index] = count
        if count >= min_count and population and frequencies[a] and frequencies[b]:
            # Compare the independence reference with Python integers first.
            # Subtracting four rounded logs can otherwise produce a tiny
            # positive score for exactly independent category pairs.
            observed = int(count) * population
            expected = int(frequencies[a]) * int(frequencies[b])
            if observed > expected:
                log_lift = math.log(observed / expected)
                scores[index] = count / (count + shrinkage) * math.tanh(log_lift)
    metadata = {"population_users": population, "n_categories": n_categories,
        "requested_pairs": int(keys.size), "positive_pairs": int(np.count_nonzero(scores)),
        "min_count": int(min_count), "shrinkage": float(shrinkage),
        "incidence_nnz": int(binary.nnz), "counting": "distinct train users; requested CSC postings only"}
    return CategoryEvidence(keys, counts, scores, metadata)


def pool_category_scores(raw_semantic, memberships, evidence):
    """Return h_ij in canonical CSR order; diagonal mass is never reallocated."""
    membership = _memberships(memberships)
    semantic = _semantic(raw_semantic, membership.shape[0])
    n_categories = membership.shape[1]
    keys, values = np.asarray(evidence.pair_keys), np.asarray(evidence.scores)
    if keys.ndim != 1 or values.shape != keys.shape or (keys.size and np.any(np.diff(keys) <= 0)):
        raise ValueError("Evidence keys must be strictly sorted and aligned with scores.")
    if keys.dtype.kind not in "iu" or (keys.size and (keys.min() < 0 or keys.max() >= n_categories ** 2)):
        raise ValueError("Evidence category keys are outside the vocabulary.")
    if keys.size and np.any(keys // n_categories >= keys % n_categories):
        raise ValueError("Evidence must encode unordered off-diagonal category pairs.")
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise ValueError("Category evidence must lie in [0, 1].")
    result = np.zeros(semantic.nnz, dtype=np.float32)
    if not keys.size or not membership.nnz:
        return result
    # Python scalar lookup avoids a NumPy binary search for every label pair.
    # At most the requested-pair cardinality is stored, never a dense C x C map.
    lookup = {int(key): float(value) for key, value in zip(keys, values) if value > 0}
    for item in range(semantic.shape[0]):
        start, end = membership.indptr[item:item + 2]
        left, left_weights = membership.indices[start:end], membership.data[start:end]
        for edge in range(semantic.indptr[item], semantic.indptr[item + 1]):
            neighbor = semantic.indices[edge]
            if neighbor == item:
                continue
            if neighbor < item:
                # On native symmetric support, reuse the already evaluated
                # opposite edge. This halves pooling work and guarantees exact
                # FP32 symmetry even for multi-label sums with different orders.
                reverse_start, reverse_end = semantic.indptr[neighbor:neighbor + 2]
                reverse_columns = semantic.indices[reverse_start:reverse_end]
                reverse = int(np.searchsorted(reverse_columns, item))
                if reverse < reverse_columns.size and reverse_columns[reverse] == item:
                    result[edge] = result[reverse_start + reverse]
                    continue
            rstart, rend = membership.indptr[neighbor:neighbor + 2]
            right, right_weights = membership.indices[rstart:rend], membership.data[rstart:rend]
            score = 0.0
            for a, weight_a in zip(left, left_weights):
                for b, weight_b in zip(right, right_weights):
                    if a == b:
                        continue
                    key = int(min(a, b)) * n_categories + int(max(a, b))
                    score += float(weight_a) * float(weight_b) * lookup.get(key, 0.0)
            result[edge] = score
    return result


def build_category_edge_scores(raw_semantic, train_edges, n_users, n_items,
                               category_data, min_count=2, shrinkage=5.0,
                               memory_budget_mib=128):
    """CPU builder with a conservative requested-pair workspace ceiling.

    The budget limits Python pair-set workspace, not total process memory:
    features, caller graphs, SciPy incidence and mapping storage remain extra.
    """
    if not math.isfinite(memory_budget_mib) or memory_budget_mib <= 0:
        raise ValueError("memory_budget_mib must be finite and positive.")
    if category_data.memberships.shape[0] != n_items:
        raise ValueError("Category catalog size mismatch.")
    max_pairs = max(1, int(memory_budget_mib * 1024 ** 2 // 256))
    requested = requested_category_pairs(raw_semantic, category_data.memberships,
        max_pairs=max_pairs, max_combinations=max_pairs * 40)
    incidence = binary_user_category_incidence(train_edges, n_users, category_data.memberships)
    evidence = category_association_scores(incidence, requested, min_count, shrinkage)
    scores = pool_category_scores(raw_semantic, category_data.memberships, evidence)
    metadata = {**evidence.metadata, "informative_coverage": category_data.audit["informative_coverage"],
        "metadata_fingerprint": category_data.metadata_fingerprint,
        "pair_workspace_budget_mib": float(memory_budget_mib),
        "active_semantic_edges": int(np.count_nonzero(scores))}
    return scores, metadata
