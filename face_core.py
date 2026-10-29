"""Face embedding storage and cosine recognition; independent of OpenCV/webcam."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re
import numpy as np

DEFAULT_THRESHOLD = 0.363  # OpenCV SFace benchmark starting point, not calibrated identity certainty.


def safe_name(name: str) -> str:
    name = name.strip()
    if not re.fullmatch(r"[\w][\w .-]{0,39}", name, re.UNICODE):
        raise ValueError("Name must be 1–40 letters/digits/spaces/dots/hyphens; no path separators")
    return name


def normalize(embedding: np.ndarray) -> np.ndarray:
    arr = np.asarray(embedding, dtype=np.float32).reshape(-1)
    magnitude = np.linalg.norm(arr)
    if arr.size == 0 or not np.isfinite(arr).all() or magnitude < 1e-8:
        raise ValueError("Invalid or empty face embedding")
    return arr / magnitude


def similarity(left: np.ndarray, right: np.ndarray) -> float:
    a, b = normalize(left), normalize(right)
    if a.size != b.size:
        raise ValueError("Embedding dimensions differ")
    return float(np.dot(a, b))

@dataclass(frozen=True)
class Match:
    name: str
    score: float
    accepted: bool


def best_match(embedding: np.ndarray, known: dict[str, np.ndarray], threshold: float = DEFAULT_THRESHOLD) -> Match:
    if not (-1 <= threshold <= 1):
        raise ValueError("Cosine threshold must be between -1 and 1")
    if not known:
        return Match("Unknown", 0.0, False)
    name, score = max(((name, similarity(embedding, emb)) for name, emb in known.items()), key=lambda x: x[1])
    return Match(name if score >= threshold else "Unknown", score, score >= threshold)


def load_gallery(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=False) as archive:
        return {safe_name(name): normalize(archive[name]) for name in archive.files}


def save_gallery(path: Path, gallery: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = {safe_name(name): normalize(feature) for name, feature in gallery.items()}
    import tempfile
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as tmp:
        temporary = Path(tmp.name)
    try:
        np.savez_compressed(temporary, **encoded)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
