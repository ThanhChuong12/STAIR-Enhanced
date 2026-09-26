"""Artifact identities, calibration and RNG-isolated controls for STAIR-RAM."""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .stair4_v4_utils import atomic_save, array_hash, identity_hash


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def implementation_hash():
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "models").glob("stair4_v5*.py"))
    paths += [root / "main_stair4_v5.py"]
    return identity_hash({str(p.relative_to(root)): file_hash(p) for p in paths})


def baseline_source_hash():
    from .stair4_v4_utils import source_hash
    root = Path(__file__).resolve().parents[1]
    return identity_hash({"reused_v4": source_hash(), "original": file_hash(root / "main.py")})


def evaluator_hash():
    import freerec
    return identity_hash({name: file_hash(inspect.getfile(value)) for name, value in (
        ("launcher", freerec.launcher.Coach), ("metrics", freerec.metrics),
        ("aggregation", freerec.utils))})


def dataset_fingerprint(dataset, mfiles):
    path = Path(dataset.path)
    names = mfiles.split(",") if isinstance(mfiles, str) else list(mfiles)
    files = [path / name for name in (*names, "train.txt", "valid.txt", "test.txt")]
    files += [p for p in path.iterdir() if p.is_file()
              and any(t in p.name.lower() for t in ("mapping", "id2", "2id"))]
    if any(not p.is_file() for p in files):
        raise FileNotFoundError("Dataset fingerprint requires all splits and modality files")
    hashes = {p.name: file_hash(p) for p in sorted(set(files))}
    return identity_hash(hashes), hashes


def write_json(path, payload):
    """Small metadata write; no unserializable tensors or nonfinite JSON values."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


@torch.no_grad()
def calibrate_scale(user_vectors, item_vectors, train_csr, *, seed, limit=100_000):
    from .stair4_v5_sampling import UniformTrainUnseenSampler
    rows, cols = train_csr.nonzero()
    if not len(rows) or limit < 1:
        raise ValueError("Calibration requires nonempty training pairs and a positive limit")
    gen = torch.Generator().manual_seed(seed)
    selected = torch.randperm(len(rows), generator=gen)[:min(limit, len(rows))]
    users = torch.as_tensor(rows, dtype=torch.long)[selected]
    positives = torch.as_tensor(cols, dtype=torch.long)[selected]
    negatives, mask = UniformTrainUnseenSampler(train_csr, seed=seed + 1).sample(users, 1)
    keep = mask[:, 0]
    users, positives, negatives = users[keep], positives[keep], negatives[keep, 0]
    if not len(users):
        raise ValueError("Calibration has no user with an unseen candidate")
    # CPU blocks avoid allocating catalog-sized scores on either device.
    u, i = user_vectors.detach().cpu(), item_vectors.detach().cpu()
    margins = []
    for start in range(0, len(users), 4096):
        batch = slice(start, start + 4096)
        margins.append(((u[users[batch]] * (i[positives[batch]] - i[negatives[batch]])).sum(-1)).abs())
    gaps = torch.cat(margins)
    # quantile uses the usual midpoint median for an even number of observations.
    sigma = max(1e-3, float(torch.quantile(gaps.float(), .5)))
    return sigma, {"sigma0": sigma, "seed": seed, "samples": len(users),
                   "sample_hash": array_hash(torch.stack([users, positives, negatives])),
                   "margin_quantiles": torch.quantile(gaps.float(), torch.tensor([0., .25, .5, .75, 1.])).tolist()}


@torch.no_grad()
def id_only_features(item_vectors, width, *, seed, names=("text", "image")):
    """Fixed semi-orthogonal lifts with no extra trainable ID parameters."""
    dimension = item_vectors.shape[1]
    if width < dimension:
        raise ValueError("The capacity-matched ID control requires feature width >= teacher dimension")
    generator = torch.Generator().manual_seed(seed)
    normalized = F.normalize(item_vectors.detach().cpu(), dim=-1, eps=1e-8)
    result = {}
    for name in names:
        lift, _ = torch.linalg.qr(torch.randn(width, dimension, generator=generator), mode="reduced")
        result[name] = normalized @ lift.T
    return result


def read_teacher(directory, *, dataset, seed, data_fingerprint, protocol):
    directory = Path(directory)
    manifest = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError("Teacher Stage A did not complete successfully")
    for key, value in (("dataset", dataset), ("seed", seed), ("data_fingerprint", data_fingerprint),
                       ("baseline_source_hash", baseline_source_hash())):
        if manifest.get(key) != value:
            raise ValueError(f"Teacher {key} differs from this run")
    if manifest.get("baseline_protocol") != protocol:
        raise ValueError("Teacher baseline/evaluation protocol differs")
    expected = manifest.get("teacher_hash")
    if not expected or file_hash(directory / "teacher.pt") != expected:
        raise ValueError("Teacher artifact checksum mismatch")
    payload = torch.load(directory / "teacher.pt", map_location="cpu", weights_only=True)
    for key in ("data_fingerprint", "comparison_id", "selected_epoch"):
        if payload.get(key) != manifest.get(key):
            raise ValueError(f"Teacher payload {key} differs from its manifest")
    return payload, manifest
