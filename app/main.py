"""HTTP API around 🍵 Matxa-TTS v2 (langtech-bsc fork of Matcha-TTS, Catalan multiaccent, graphemes)."""
from __future__ import annotations

import asyncio
import dataclasses
import logging
import os
from contextlib import asynccontextmanager
from functools import partial
from importlib.metadata import PackageNotFoundError, version

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, JSONResponse

from app.engine import ModelLoadError, SynthesisError, TTSEngine, encode
from app.registry import available_models, default_model
from app.schemas import (
    HealthResponse,
    ModelInfo,
    ModelsResponse,
    SpeakerRange,
    TokenizeRequest,
    TTSRequest,
    VoiceInfo,
)

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("matcha-api")

engine = TTSEngine()
# Serialise work so a burst of requests queues instead of thrashing the CPU.
_slots = asyncio.Semaphore(int(os.environ.get("MATCHA_MAX_CONCURRENCY", "1")))


def _matcha_version() -> str:
    try:
        return version("matcha-tts")
    except PackageNotFoundError:  # pragma: no cover - installed from git in the image
        return "unknown"


@asynccontextmanager
async def lifespan(_: FastAPI):
    if os.environ.get("MATCHA_PRELOAD", "1") == "1":
        name = default_model()
        log.info("Preloading %s on %s", name, engine.device)
        try:
            await run_in_threadpool(engine.warm, name)
        except Exception:  # noqa: BLE001 - server should still boot and report /health
            log.exception("Preloading %s failed; it will be retried on first request", name)
    yield


app = FastAPI(
    title="Matcha-TTS API",
    description="REST wrapper around Matxa-TTS v2, the grapheme-based Catalan multiaccent Matcha-TTS model.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(SynthesisError)
async def _synthesis_error(_, exc: SynthesisError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(ModelLoadError)
async def _model_load_error(_, exc: ModelLoadError):
    log.error("model load failed: %s", exc)
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.get("/health", response_model=HealthResponse, tags=["meta"])
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        device=str(engine.device),
        default_model=default_model(),
        loaded_models=engine.loaded(),
        matcha_version=_matcha_version(),
    )


@app.get("/v1/models", response_model=ModelsResponse, tags=["meta"])
async def models() -> ModelsResponse:
    loaded = set(engine.loaded())
    return ModelsResponse(
        default=default_model(),
        device=str(engine.device),
        models=[
            ModelInfo(
                name=spec.name,
                vocoder=spec.vocoder,
                speaking_rate=spec.speaking_rate,
                multispeaker=spec.multispeaker,
                speakers=SpeakerRange(n_speakers=spec.n_speakers, default=spec.default_speaker),
                voices=[VoiceInfo(**dataclasses.asdict(v)) for v in spec.voices],
                loaded=spec.name in loaded,
                description=spec.description,
            )
            for spec in sorted(available_models().values(), key=lambda s: s.name)
        ],
    )


@app.post("/v1/tokenize", tags=["tts"])
async def tokenize(req: TokenizeRequest) -> dict:
    return await run_in_threadpool(engine.tokenise, req.text, req.model)


async def _synthesise(req: TTSRequest) -> Response:
    async with _slots:
        result = await run_in_threadpool(
            partial(
                engine.synthesise,
                text=req.text,
                model=req.model,
                speaker=req.speaker,
                voice=req.voice,
                temperature=req.temperature,
                speaking_rate=req.speaking_rate,
                steps=req.steps,
                sway_sampling=req.sway_sampling,
                denoiser_strength=req.denoiser_strength,
            )
        )
    payload, content_type = encode(result, req.format)
    log.info(
        "synthesised model=%s voice=%s chars=%d audio=%.2fs synth=%.2fs rtf=%.3f",
        result.model,
        result.voice or result.speaker,
        len(req.text),
        result.audio_seconds,
        result.synth_seconds,
        result.rtf,
    )
    return Response(
        content=payload,
        media_type=content_type,
        headers={
            "X-Matcha-Model": result.model,
            "X-Matcha-Speaker": "" if result.speaker is None else str(result.speaker),
            "X-Matcha-Voice": result.voice or "",
            "X-Matcha-Audio-Seconds": f"{result.audio_seconds:.3f}",
            "X-Matcha-Synthesis-Seconds": f"{result.synth_seconds:.3f}",
            "X-Matcha-RTF": f"{result.rtf:.4f}",
            "Content-Disposition": f'inline; filename="speech.{req.format}"',
        },
    )


@app.post(
    "/v1/tts",
    tags=["tts"],
    responses={200: {"content": {"audio/wav": {}}, "description": "Synthesised audio."}},
    response_class=Response,
)
async def tts(req: TTSRequest) -> Response:
    return await _synthesise(req)


@app.get("/v1/tts", tags=["tts"], response_class=Response)
async def tts_get(
    text: str = Query(..., description="Text to synthesise."),
    model: str | None = None,
    speaker: int | None = None,
    voice: str | None = None,
    steps: int = 20,
    temperature: float = 0.7,
    speaking_rate: float | None = None,
    sway_sampling: float | None = -1.0,
    denoiser_strength: float = 0.0025,
    format: str = "wav",  # noqa: A002 - matches the JSON field name
) -> Response:
    """Convenience variant so a browser or <audio src> can hit the API directly."""
    return await _synthesise(
        TTSRequest(
            text=text,
            model=model,
            speaker=speaker,
            voice=voice,
            steps=steps,
            temperature=temperature,
            speaking_rate=speaking_rate,
            sway_sampling=sway_sampling,
            denoiser_strength=denoiser_strength,
            format=format,
        )
    )


@app.get("/", include_in_schema=False)
async def index() -> HTMLResponse:
    return HTMLResponse(
        """<!doctype html><title>Matxa-TTS API</title>
        <h1>🍵 Matxa-TTS API</h1>
        <p>Interactive docs: <a href="/docs">/docs</a> &middot;
           models: <a href="/v1/models">/v1/models</a> &middot;
           health: <a href="/health">/health</a></p>
        <form onsubmit="go(event)">
          <input id="t" size="60" value="Bon dia Manel, avui anem a la muntanya.">
          <select id="v"></select>
          <button>Speak</button>
        </form>
        <audio id="a" controls></audio>
        <script>
        fetch('/v1/models').then(r=>r.json()).then(d=>{
          const m=d.models.find(m=>m.name===d.default), v=document.getElementById('v');
          if(!m||!m.voices.length){v.hidden=true;return;}
          for(const x of m.voices){
            const o=new Option(x.name+' ('+x.accent+')',x.id);o.selected=x.id===m.speakers.default;v.add(o);}
        });
        function go(e){e.preventDefault();
          const v=document.getElementById('v'),a=document.getElementById('a');
          a.src='/v1/tts?text='+encodeURIComponent(document.getElementById('t').value)
            +(v.value?'&speaker='+v.value:'');
          a.play();}
        </script>"""
    )


@app.get("/healthz", include_in_schema=False)
async def healthz() -> dict:
    if not engine.loaded() and os.environ.get("MATCHA_PRELOAD", "1") == "1":
        raise HTTPException(status_code=503, detail="model not loaded yet")
    return {"status": "ok"}
