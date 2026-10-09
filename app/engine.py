"""Model loading and synthesis.

Mirrors what the fork's ``matxa_graphemes_alvocat_inference.py`` does, minus the
argparse/file plumbing, and keeps loaded checkpoints resident between requests.
"""
from __future__ import annotations

import io
import logging
import os
import threading
import time
from dataclasses import dataclass

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import hf_hub_download
from matcha.models.matcha_tts import MatchaTTS
from matcha.text import _symbol_to_id, cleaners, sequence_to_text, text_to_sequence
from vocos import Vocos

from app.registry import (
    CLEANER,
    HUB_FILES,
    SAMPLE_RATE,
    VOCODERS,
    ModelSpec,
    available_models,
    default_model,
)

log = logging.getLogger("matcha-api")

FORMATS = {
    "wav": ("WAV", "audio/wav", "PCM_16"),
    "flac": ("FLAC", "audio/flac", "PCM_16"),
    "ogg": ("OGG", "audio/ogg", "VORBIS"),
}


class SynthesisError(ValueError):
    """Bad request-level input (unknown model, out-of-range speaker, ...)."""


class ModelLoadError(RuntimeError):
    """A checkpoint could not be downloaded or loaded (server-side problem)."""


@dataclass
class Synthesis:
    audio: np.ndarray
    sample_rate: int
    text: str
    audio_seconds: float
    synth_seconds: float
    model: str
    speaker: int | None
    voice: str | None

    @property
    def rtf(self) -> float:
        return self.synth_seconds / self.audio_seconds if self.audio_seconds else 0.0


def resolve_device(requested: str | None = None) -> torch.device:
    name = requested or os.environ.get("MATCHA_DEVICE", "auto")
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def _fetch(repo: str, revision: str | None) -> None:
    try:
        for filename in HUB_FILES:
            hf_hub_download(repo_id=repo, filename=filename, revision=revision)
    except Exception as exc:  # noqa: BLE001 - surfaced as 503, not a traceback
        raise ModelLoadError(f"Could not download {repo}: {exc}") from exc


def _vocoder_repo(vocoder: str) -> tuple[str, str | None]:
    if vocoder not in VOCODERS:
        raise SynthesisError(f"Unknown vocoder '{vocoder}'. Available: {', '.join(VOCODERS)}.")
    return VOCODERS[vocoder]


def download_model(spec: ModelSpec) -> None:
    """Fetch everything ``spec`` needs into the hub cache (used by the prefetch script)."""
    if spec.hf_repo is not None:
        _fetch(spec.hf_repo, spec.hf_revision)
    _fetch(*_vocoder_repo(spec.vocoder))


class TTSEngine:
    def __init__(self, device: torch.device | None = None) -> None:
        self.device = device or resolve_device()
        self._models: dict[str, MatchaTTS] = {}
        self._vocoders: dict[str, Vocos] = {}
        self._load_lock = threading.Lock()
        # Inference is CPU/GPU bound and the graph is not reentrant-safe; one
        # request at a time per process keeps latency predictable.
        self._infer_lock = threading.Lock()

    # ----------------------------------------------------------------- loading
    def spec(self, model: str | None) -> ModelSpec:
        models = available_models()
        name = model or default_model()
        if name not in models:
            raise SynthesisError(f"Unknown model '{name}'. Available: {', '.join(sorted(models))}.")
        return models[name]

    def loaded(self) -> list[str]:
        return sorted(self._models)

    def _matcha(self, spec: ModelSpec) -> MatchaTTS:
        if spec.name not in self._models:
            with self._load_lock:
                if spec.name not in self._models:
                    source = spec.checkpoint or spec.hf_repo
                    log.info("Loading Matcha checkpoint %s", source)
                    try:
                        if spec.checkpoint is not None:
                            model = MatchaTTS.load_from_checkpoint(str(spec.checkpoint), map_location=self.device)
                        else:
                            model = MatchaTTS.from_pretrained(
                                spec.hf_repo, token_hf=None, device=self.device, revision=spec.hf_revision
                            )
                    except Exception as exc:  # noqa: BLE001 - surfaced as 503, not a traceback
                        raise ModelLoadError(f"Could not load checkpoint {source}: {exc}") from exc
                    model.to(self.device).eval()
                    self._models[spec.name] = model
        return self._models[spec.name]

    def _vocoder(self, name: str) -> Vocos:
        if name not in self._vocoders:
            with self._load_lock:
                if name not in self._vocoders:
                    repo, revision = _vocoder_repo(name)
                    log.info("Loading vocoder %s", repo)
                    try:
                        vocos = Vocos.from_pretrained(repo, device=self.device, revision=revision)
                    except Exception as exc:  # noqa: BLE001 - surfaced as 503, not a traceback
                        raise ModelLoadError(f"Could not load vocoder {repo}: {exc}") from exc
                    self._vocoders[name] = vocos.to(self.device).eval()
        return self._vocoders[name]

    def warm(self, model: str | None = None) -> None:
        spec = self.spec(model)
        self._matcha(spec)
        self._vocoder(spec.vocoder)

    # --------------------------------------------------------------- front-end
    def tokenise(self, text: str, model: str | None = None) -> dict:
        spec = self.spec(model)
        ids = _to_ids(text)
        return {"model": spec.name, "text": sequence_to_text(ids), "ids": ids}

    @staticmethod
    def _speaker_id(spec: ModelSpec, speaker: int | None, voice: str | None) -> int | None:
        if voice is not None:
            match = next((v for v in spec.voices if v.name == voice.strip().lower()), None)
            if match is None:
                names = ", ".join(v.name for v in spec.voices) or "none"
                raise SynthesisError(f"Unknown voice '{voice}' for '{spec.name}'. Available: {names}.")
            if speaker not in (None, match.id):
                raise SynthesisError(f"Voice '{match.name}' is speaker {match.id}, not {speaker}; send only one.")
            return match.id
        if not spec.multispeaker:
            if speaker not in (None, spec.default_speaker):
                raise SynthesisError(f"Model '{spec.name}' is single-speaker; drop the 'speaker' field.")
            return None
        sid = spec.default_speaker if speaker is None else speaker
        if not 0 <= sid < spec.n_speakers:
            raise SynthesisError(f"Speaker id must be in [0, {spec.n_speakers - 1}] for '{spec.name}'.")
        return sid

    # -------------------------------------------------------------- synthesis
    @torch.inference_mode()
    def synthesise(
        self,
        text: str,
        model: str | None = None,
        speaker: int | None = None,
        voice: str | None = None,
        temperature: float = 0.7,
        speaking_rate: float | None = None,
        steps: int = 20,
        sway_sampling: float | None = -1.0,
        denoiser_strength: float = 0.0025,
    ) -> Synthesis:
        text = text.strip()
        if not text:
            raise SynthesisError("'text' must not be empty.")

        spec = self.spec(model)
        sid = self._speaker_id(spec, speaker, voice)
        length_scale = spec.speaking_rate if speaking_rate is None else speaking_rate
        spks = None if sid is None else torch.tensor([sid], device=self.device, dtype=torch.long)

        matcha = self._matcha(spec)
        vocoder = self._vocoder(spec.vocoder)

        ids = _to_ids(text)
        x = torch.tensor(_intersperse(ids, 0), dtype=torch.long, device=self.device)[None]
        x_lengths = torch.tensor([x.shape[-1]], dtype=torch.long, device=self.device)

        started = time.perf_counter()
        with self._infer_lock:
            output = matcha.synthesise(
                x,
                x_lengths,
                n_timesteps=steps,
                temperature=temperature,
                spks=spks,
                length_scale=length_scale,
                sway_sampling_coef=sway_sampling,
            )
            audio = _vocode(vocoder, output["mel"], denoiser_strength).clamp(-1, 1)
            waveform = audio.float().cpu().squeeze().numpy()
        elapsed = time.perf_counter() - started

        return Synthesis(
            audio=waveform,
            sample_rate=SAMPLE_RATE,
            text=sequence_to_text(ids),
            audio_seconds=len(waveform) / SAMPLE_RATE,
            synth_seconds=elapsed,
            model=spec.name,
            speaker=sid,
            voice=getattr(spec.voice(sid), "name", None),
        )


def _to_ids(text: str) -> list[int]:
    # The model reads graphemes directly, so anything outside its symbol table
    # (digits, %, &, ...) has no embedding and must be spelled out by the caller.
    unknown = sorted({c for c in getattr(cleaners, CLEANER)(text) if c not in _symbol_to_id})
    if unknown:
        raise SynthesisError(
            f"Unsupported character(s) {' '.join(map(repr, unknown))}: "
            "write numbers and symbols out in words, the model reads plain Catalan text."
        )
    return text_to_sequence(text, [CLEANER])


def _vocode(vocoder: Vocos, mel: torch.Tensor, denoiser_strength: float) -> torch.Tensor:
    """Mel -> waveform, subtracting the vocoder's bias spectrum as the fork's script does."""
    audio, spec = vocoder.decode(mel)
    if denoiser_strength <= 0:
        return audio
    _, bias = vocoder.decode(torch.zeros_like(mel))
    magnitude = (spec.abs() - bias.abs() * denoiser_strength).clamp(min=0.0)
    return vocoder.head.istft(torch.polar(magnitude, spec.angle()))


def _intersperse(seq: list[int], item: int) -> list[int]:
    """Same as matcha.utils.utils.intersperse, inlined to avoid its heavy imports."""
    result = [item] * (len(seq) * 2 + 1)
    result[1::2] = seq
    return result


def encode(result: Synthesis, fmt: str = "wav") -> tuple[bytes, str]:
    """Serialise a synthesis to bytes, returning (payload, content-type)."""
    if fmt not in FORMATS:
        raise SynthesisError(f"Unsupported format '{fmt}'. Allowed: {', '.join(FORMATS)}.")
    sf_format, content_type, subtype = FORMATS[fmt]
    buffer = io.BytesIO()
    sf.write(buffer, result.audio, result.sample_rate, format=sf_format, subtype=subtype)
    return buffer.getvalue(), content_type
