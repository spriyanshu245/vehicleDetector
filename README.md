# vehicleDetector

Upload a vehicle photo, get the number plate read out. Contour-based plate
localization (OpenCV) + text recognition (EasyOCR), served as a JSON API
behind a single-page drag-and-drop UI.

## How it works

1. Image is decoded in memory (never written to disk) and downscaled for detection.
2. Bilateral filter → Canny edges → contour search for wide quadrilateral
   candidates (plate-shaped), de-duplicated.
3. Top candidates are OCR'd with EasyOCR (English, CPU); the highest-confidence
   alphanumeric read wins.
4. API returns `{plate_text, confidence, boxes, match, image_width, image_height}`;
   the UI draws the boxes on a canvas overlay.

## Run locally

```sh
pip install -r requirements.txt
python app.py            # http://localhost:8080
```

First request downloads ~100MB of EasyOCR weights to `~/.EasyOCR/model`
(or `$EASYOCR_MODEL_DIR`) and takes up to a minute; later requests are fast.

## API

- `GET /` — the UI
- `GET /health` — `{"status": "ok", "reader_ready": bool}` (warmup pings)
- `POST /api/detect` — multipart form, field `image` (png/jpg/jpeg/webp, ≤16MB)

```sh
curl -X POST -F "image=@car.jpg" http://localhost:8080/api/detect
```

## Deploy — Heroku container (recommended)

The Dockerfile is multi-stage and **bakes the EasyOCR weights into the image**,
so dyno restarts never re-download them (Heroku's disk is ephemeral).

```sh
heroku create vehicle-detector-app --stack container
heroku config:set SECRET_KEY="$(openssl rand -hex 32)" LOG_LEVEL=INFO
heroku stack:set container
git push heroku modernize/vehicle-detector:main
```

`heroku.yml` alternative:

```yaml
build:
  docker:
    web: Dockerfile
```

### Required env vars

| Var | Purpose |
|---|---|
| `SECRET_KEY` | Flask secret key (generate a random one) |
| `PORT` | set automatically by Heroku |
| `EASYOCR_MODEL_DIR` | `/app/models` (already set in the image) |
| `LOG_LEVEL` | optional, defaults to `INFO` |
| `FLASK_DEBUG` | optional, `1` enables debug (never in prod) |

### Cold-start tradeoff

- **Baked weights (this Dockerfile):** slower first *build*, instant cold starts —
  the model is already on disk when the container boots.
- **Download-on-boot:** smaller image, but every dyno restart re-downloads
  ~100MB before the first detection works.

The OCR reader is a lazy singleton: with gunicorn `--workers 1 --threads 4`
(one worker — EasyOCR + torch use ~300–400MB RAM) the first request after a
cold start initializes it once; concurrent first-requests are serialized by a
lock. Use `GET /health` from an uptime pinger to keep it warm.

## Deploy — classic Heroku buildpack

```sh
heroku create vehicle-detector-app
heroku buildpacks:set heroku/python
heroku config:set SECRET_KEY="$(openssl rand -hex 32)"
git push heroku modernize/vehicle-detector:main
```

`requirements.txt` pins CPU-only torch via the PyTorch wheel index so the slug
stays small. Note: on the buildpack path the EasyOCR weights download on first
request (ephemeral disk) — prefer the container path above.
