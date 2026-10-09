"""Catalogue of checkpoints the API can serve."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

SAMPLE_RATE = 22050

# Vocos vocoders, loaded from the Hugging Face hub: name -> (repo, revision).
VOCODERS = {
    "alvocat": ("projecte-aina/alvocat-vocos-22khz", "fd94916a78d7f8b9cff2faf688c3da0866ba61cf"),
}

# Files MatchaTTS.from_pretrained / Vocos.from_pretrained read from a hub repo.
HUB_FILES = ("config.yaml", "pytorch_model.bin")

# v2 is grapheme-based: text is only lowercased and whitespace-collapsed.
CLEANER = "basic_cleaners"


@dataclass(frozen=True)
class Voice:
    id: int
    name: str
    accent: str
    dataset: str


@dataclass(frozen=True)
class ModelSpec:
    name: str
    vocoder: str
    speaking_rate: float
    n_speakers: int = 1
    default_speaker: int | None = None
    hf_repo: str | None = None  # None => checkpoint is mounted, not downloadable
    hf_revision: str | None = None
    checkpoint: Path | None = None
    description: str = ""
    voices: tuple[Voice, ...] = ()

    @property
    def multispeaker(self) -> bool:
        return self.n_speakers > 1

    def voice(self, speaker: int | None) -> Voice | None:
        return next((v for v in self.voices if v.id == speaker), None)


BUILTIN_MODELS: dict[str, ModelSpec] = {
    "matxa_v2_multiaccent": ModelSpec(
        name="matxa_v2_multiaccent",
        vocoder="alvocat",
        speaking_rate=0.9,
        n_speakers=16,
        default_speaker=3,
        hf_repo="BSC-LT/matxa-tts-v2-ca-multiaccent-graphemes",
        hf_revision="00f32fa221aef52e0146422935bf87fc7d71db6f",
        description="Matxa-TTS v2 Catalan multiaccent (graphemes): 16 voices across 5 dialects.",
        # Ids differ from v1 (there quim was 0 and olga 1, etc.).
        voices=(
            Voice(0, "olga", "balear", "LaFresCat"),
            Voice(1, "quim", "balear", "LaFresCat"),
            Voice(2, "elia", "central", "LaFresCat"),
            Voice(3, "grau", "central", "LaFresCat"),
            Voice(4, "emma", "nord-occidental", "LaFresCat"),
            Voice(5, "pere", "nord-occidental", "LaFresCat"),
            Voice(6, "gina", "valencia", "LaFresCat"),
            Voice(7, "lluc", "valencia", "LaFresCat"),
            Voice(8, "arnau", "valencia", "ReFresCat"),
            Voice(9, "berta", "valencia", "ReFresCat"),
            Voice(10, "bm", "balear", "EnVeuAlta"),
            Voice(11, "cf", "central", "EnVeuAlta"),
            Voice(12, "cm", "central", "EnVeuAlta"),
            Voice(13, "estel", "nord-occidental", "ReFresCat"),
            Voice(14, "laura", "septentrional", "ReFresCat"),
            Voice(15, "jordi", "septentrional", "ReFresCat"),
        ),
    ),
}


def data_dir() -> Path:
    """Where checkpoints live. Matcha reads MATCHA_HOME itself, so honour it."""
    return Path(os.environ.get("MATCHA_HOME", "/models")).expanduser()


def custom_dir() -> Path:
    return data_dir() / "custom"


def discover_custom_models() -> dict[str, ModelSpec]:
    """Register every ``*.ckpt`` dropped into ``$MATCHA_HOME/custom``.

    Lets you serve your own Matxa checkpoints without rebuilding the image.
    Per-checkpoint overrides can be set with env vars named
    ``MATCHA_<NAME>_SPEAKERS`` / ``_RATE`` / ``_VOCODER``.
    """
    models: dict[str, ModelSpec] = {}
    root = custom_dir()
    if not root.is_dir():
        return models

    for ckpt in sorted(root.glob("*.ckpt")):
        name = ckpt.stem
        prefix = f"MATCHA_{name.upper().replace('-', '_')}_"
        n_speakers = int(os.environ.get(prefix + "SPEAKERS", "1"))
        models[name] = ModelSpec(
            name=name,
            vocoder=os.environ.get(prefix + "VOCODER", "alvocat"),
            speaking_rate=float(os.environ.get(prefix + "RATE", "1.0")),
            n_speakers=n_speakers,
            default_speaker=0 if n_speakers > 1 else None,
            checkpoint=ckpt,
            description=f"Custom checkpoint mounted at {ckpt}.",
        )
    return models


def available_models() -> dict[str, ModelSpec]:
    models = dict(BUILTIN_MODELS)
    models.update(discover_custom_models())
    return models


def default_model() -> str:
    return os.environ.get("MATCHA_DEFAULT_MODEL", "matxa_v2_multiaccent")
