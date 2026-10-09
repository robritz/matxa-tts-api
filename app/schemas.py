"""Request/response models for the API."""
from __future__ import annotations

import os

from pydantic import BaseModel, Field

MAX_TEXT_CHARS = int(os.environ.get("MATCHA_MAX_TEXT_CHARS", "2000"))


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_CHARS, description="Text to synthesise.")
    model: str | None = Field(None, description="Checkpoint name; defaults to MATCHA_DEFAULT_MODEL.")
    speaker: int | None = Field(None, description="Speaker id for multi-speaker checkpoints.")
    voice: str | None = Field(None, description="Voice name instead of a speaker id, e.g. 'quim'.")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Variance of the x0 noise.")
    speaking_rate: float | None = Field(
        None, gt=0.0, le=5.0, description="Length scale; higher is slower. Defaults to the model's own value."
    )
    steps: int = Field(20, ge=1, le=100, description="Euler ODE solver steps.")
    sway_sampling: float | None = Field(
        -1.0, ge=-1.0, le=1.0, description="Sway sampling coefficient for the ODE time grid; null disables it."
    )
    denoiser_strength: float = Field(0.0025, ge=0.0, le=1.0, description="Vocoder bias denoiser; 0 disables it.")
    format: str = Field("wav", description="Audio container: wav, flac or ogg.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "Bon dia Manel, avui anem a la muntanya.",
                "model": "matxa_v2_multiaccent",
                "voice": "grau",
            }
        }
    }


class TokenizeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_CHARS)
    model: str | None = None


class SpeakerRange(BaseModel):
    n_speakers: int
    default: int | None


class VoiceInfo(BaseModel):
    id: int
    name: str
    accent: str
    dataset: str


class ModelInfo(BaseModel):
    name: str
    vocoder: str
    speaking_rate: float
    multispeaker: bool
    speakers: SpeakerRange
    voices: list[VoiceInfo]
    loaded: bool
    description: str


class ModelsResponse(BaseModel):
    default: str
    device: str
    models: list[ModelInfo]


class HealthResponse(BaseModel):
    status: str
    device: str
    default_model: str
    loaded_models: list[str]
    matcha_version: str
