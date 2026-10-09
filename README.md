# 🍵 Matxa-TTS v2 API (dockerized)

A small FastAPI service around
[Matxa-TTS v2 Catalan multiaccent (graphemes)](https://huggingface.co/BSC-LT/matxa-tts-v2-ca-multiaccent-graphemes)
— the `v2-graphemes` branch of [langtech-bsc/Matcha-TTS](https://github.com/langtech-bsc/Matcha-TTS/tree/v2-graphemes)
with the [🥑 alVoCat](https://huggingface.co/projecte-aina/alvocat-vocos-22khz) Vocos vocoder.
Sixteen voices across five Catalan dialects, no phonemizer needed. Checkpoints
stay resident between requests, so a request costs one forward pass instead of
a cold model load.

## Quick start

```bash
docker compose up --build
```

First boot downloads the model (~213 MB) plus the alVoCat vocoder (~54 MB) from
the Hugging Face hub into the `matcha-models` volume, then preloads them.
Afterwards:

```bash
curl -X POST localhost:8000/v1/tts \
  -H 'content-type: application/json' \
  -d '{"text":"Bon dia Manel, avui anem a la muntanya.","voice":"grau"}' \
  -o speech.wav
```

Open <http://localhost:8000/> for a one-box demo page, or
<http://localhost:8000/docs> for the generated OpenAPI docs.

## Endpoints

| Method | Path             | Purpose                                              |
| ------ | ---------------- | ---------------------------------------------------- |
| POST   | `/v1/tts`        | Synthesise; returns `audio/wav` (or flac/ogg) bytes  |
| GET    | `/v1/tts?text=…` | Same, for `<audio src>` / browser use                |
| POST   | `/v1/tokenize`   | Normalised text + symbol ids (debugging)             |
| GET    | `/v1/models`     | Available checkpoints, voices, defaults              |
| GET    | `/health`        | Device, default model, loaded models, matcha version |
| GET    | `/healthz`       | Liveness/readiness probe used by `HEALTHCHECK`       |

### `POST /v1/tts` body

| Field               | Default                | Notes                                            |
| ------------------- | ---------------------- | ------------------------------------------------ |
| `text`              | –                      | Required, ≤ `MATCHA_MAX_TEXT_CHARS` characters    |
| `model`             | `matxa_v2_multiaccent` | Any name from `/v1/models`                        |
| `voice`             | `grau`                 | Voice name from the table below                   |
| `speaker`           | `3`                    | Speaker id, alternative to `voice`                |
| `steps`             | `20`                   | Euler ODE steps; fewer = faster, rougher          |
| `temperature`       | `0.7`                  | Variance of the x0 noise                          |
| `speaking_rate`     | `0.9`                  | Length scale; higher = slower                     |
| `sway_sampling`     | `-1.0`                 | Sway-sampling coefficient; `null` disables it     |
| `denoiser_strength` | `0.0025`               | Vocoder bias denoiser; `0` disables it            |
| `format`            | `wav`                  | `wav` \| `flac` \| `ogg`                          |

### Text input

The model reads graphemes directly: text is lowercased and whitespace is
collapsed, nothing else. **Numbers and symbols must be written out in words**
(`quaranta-dos`, not `42`) — characters outside the model's alphabet return
`400` naming the offending characters.

### Voices

| `speaker` | `voice` | Accent          | Dataset   |
| --------- | ------- | --------------- | --------- |
| 0         | `olga`  | balear          | LaFresCat |
| 1         | `quim`  | balear          | LaFresCat |
| 2         | `elia`  | central         | LaFresCat |
| 3         | `grau`  | central         | LaFresCat |
| 4         | `emma`  | nord-occidental | LaFresCat |
| 5         | `pere`  | nord-occidental | LaFresCat |
| 6         | `gina`  | valencia        | LaFresCat |
| 7         | `lluc`  | valencia        | LaFresCat |
| 8         | `arnau` | valencia        | ReFresCat |
| 9         | `berta` | valencia        | ReFresCat |
| 10        | `bm`    | balear          | EnVeuAlta |
| 11        | `cf`    | central         | EnVeuAlta |
| 12        | `cm`    | central         | EnVeuAlta |
| 13        | `estel` | nord-occidental | ReFresCat |
| 14        | `laura` | septentrional   | ReFresCat |
| 15        | `jordi` | septentrional   | ReFresCat |

Speaker ids are not the same as in Matxa v1 (e.g. `quim` was 0 there); prefer
`voice` names.

Responses carry timing metadata as headers: `X-Matcha-Model`,
`X-Matcha-Speaker`, `X-Matcha-Voice`, `X-Matcha-Audio-Seconds`, `X-Matcha-Synthesis-Seconds`,
`X-Matcha-RTF`.

Examples:

```bash
# Valencian voice, ogg out
curl "localhost:8000/v1/tts?text=La%20seva%20gerra%20sembla%20molt%20antiga.&voice=gina&format=ogg" -o gina.ogg

# slower, higher quality
curl -X POST localhost:8000/v1/tts -H 'content-type: application/json' \
  -d '{"text":"A poc a poc.","steps":40,"speaking_rate":1.15}' -o slow.wav

# what does the model actually see?
curl -X POST localhost:8000/v1/tokenize -H 'content-type: application/json' \
  -d '{"text":"Tinc quaranta-dos anys."}'
```

Audio is mono 22.05 kHz. Invalid input returns `400`/`422` with a `detail`
message; a checkpoint that cannot be loaded returns `503`.

## Configuration

Environment variables (see `docker-compose.yml`):

| Variable                 | Default            | Meaning                                       |
| ------------------------ | ------------------ | --------------------------------------------- |
| `MATCHA_DEFAULT_MODEL`   | `matxa_v2_multiaccent` | Model used when a request omits `model`    |
| `MATCHA_DEVICE`          | `auto`             | `auto` \| `cpu` \| `cuda`                      |
| `MATCHA_PRELOAD`         | `1`                | Load the default model at startup              |
| `MATCHA_MAX_CONCURRENCY` | `1`                | Concurrent syntheses per container             |
| `MATCHA_MAX_TEXT_CHARS`  | `2000`             | Request size cap                               |
| `MATCHA_HOME`            | `/models`          | Volume root; custom checkpoints in `custom/`   |
| `HF_HOME`                | `/models/hf`       | Hugging Face cache holding the downloaded weights |
| `OMP_NUM_THREADS`        | `4`                | CPU threads per synthesis                      |
| `PORT`                   | `8000`             | Listen port                                    |

Synthesis holds a process-wide lock, so `MATCHA_MAX_CONCURRENCY` mainly bounds
the queue; scale out with replicas (they can share the model volume) rather than
up, or give each container more `OMP_NUM_THREADS`.

### Baking checkpoints into the image

Downloading on first boot keeps the image small; baking makes containers
start cold-free (useful for autoscaled deployments):

```bash
docker build --build-arg PREFETCH_MODELS="matxa_v2_multiaccent" -t matcha-tts-api .
# or, into an existing volume:
docker compose run --rm api python scripts/prefetch_models.py matxa_v2_multiaccent
```

Baked checkpoints live in `/models`, so they survive a *named* volume mount
(Docker seeds it from the image) but are hidden by a bind mount there.

### Custom checkpoints

Drop any Matcha `*.ckpt` into `./custom-checkpoints/` (mounted at
`/models/custom`) and it is served under its file name — this is how to use your
own model trained on the `v2-graphemes` branch:

```bash
cp my_voice.ckpt custom-checkpoints/
curl -X POST localhost:8000/v1/tts -H 'content-type: application/json' \
  -d '{"text":"Hola!","model":"my_voice"}' -o out.wav
```

Defaults assume single speaker, the `alvocat` vocoder and rate `1.0`.
Override per checkpoint with `MATCHA_<NAME>_SPEAKERS`, `_RATE`, `_VOCODER`
(e.g. `MATCHA_MY_VOICE_SPEAKERS=4`). Vocoders are listed in
`app/registry.VOCODERS`.

### GPU

```bash
docker build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 -t matcha-tts-api:cuda .
```

Then uncomment the `deploy.resources` block in `docker-compose.yml`. The
service picks CUDA automatically when it is visible (`MATCHA_DEVICE=auto`).

## Hosting

Anything that runs a Docker container with about 2 vCPUs and 4 GB of RAM
works; no GPU is needed. These are suggestions, not tested deployments — the
sizing is an estimate from running the container locally on CPU.

| Option                                    | Good for                                        | What to change                                                                                                   |
| ----------------------------------------- | ----------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| Hugging Face Spaces (Docker SDK)          | Public demo, free CPU tier                      | Set `PORT=7860` and bake the weights in with `PREFETCH_MODELS`. Free Spaces sleep when idle.                      |
| A small VPS (Hetzner, DigitalOcean, etc.) | Always-on, cheapest per month, full control     | Nothing: `docker compose up -d` works as is. Put Caddy or nginx in front for HTTPS.                               |
| Google Cloud Run                          | Pay per use, scales to zero                     | Bake the weights into the image (no persistent volume). Expect a cold start of 15 s or more while the model loads. |
| Fly.io / Railway / Render                 | Simple deploys straight from the Dockerfile     | Attach a volume at `/models`, or bake the weights in. The smallest instance tiers are too small for PyTorch.       |

Before exposing it publicly:

- **No authentication.** The API has no auth or rate limiting, so anyone with
  the URL can use it. Add an API key check or put it behind a proxy.
- **Licence.** Matxa v2 is non-commercial only, and hosting it for others means
  passing its use restrictions on to your users (see [License](#license)).

## Layout

```
Dockerfile              multi-stage build (venv in builder, no toolchain at runtime)
docker-compose.yml       service + model volume + custom-checkpoint mount
requirements.txt         pinned inference + API deps (Matcha installed --no-deps)
app/main.py              FastAPI routes
app/engine.py            checkpoint loading, synthesis, audio encoding
app/registry.py          model catalogue + custom-checkpoint discovery
app/schemas.py           request/response models
scripts/prefetch_models.py  checkpoint downloader
```

Notes on the build: Matcha's own `requirements.txt` is training-oriented
(gradio, notebook, optuna) and conflicts with a current FastAPI stack, so the
package is installed with `--no-deps` against the pinned set in
`requirements.txt`. The `v2-graphemes` branch ships its own `vocos` package, so
no separate Vocos install is needed. `MATCHA_REF` in the Dockerfile pins the
upstream commit; the Hugging Face revisions are pinned in `app/registry.py`.
The only system packages needed at runtime are `libsndfile1` and `libgomp1`.

Long inputs are synthesised in one pass — split paragraphs into sentences
client-side for predictable memory and latency.

## License

This wrapper is MIT. Matxa-TTS v2 is released under a custom non-commercial
OpenRAIL-M licence: research and non-profit use only, with use restrictions
that you must pass on to your own users if you host it for others — read
[LICENSE.md](https://huggingface.co/BSC-LT/matxa-tts-v2-ca-multiaccent-graphemes/blob/main/LICENSE.md).
Commercial use needs a licence from the voice artists. alVoCat keeps its own
license.
