import base64
import tempfile
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from .infer import generate_talking_head  # ← relative import
import os

app = FastAPI()


@app.get("/health")
async def health():
    return {"ok": True}


@app.post("/generateVideo")
async def generate_video(
    face: UploadFile = File(...),
    audio: UploadFile = File(...),
    x_api_key: str = Header(None),
):
    # simple header check
    if x_api_key != os.environ.get("SADTALKER_API_KEY", "supersecret-local"):
        raise HTTPException(status_code=401, detail="unauthorized")

    # make a temp working dir per request
    workdir = Path(tempfile.mkdtemp(prefix="sadtalker_"))

    face_path = workdir / "face.png"
    audio_path = workdir / "audio.wav"

    face_path.write_bytes(await face.read())
    audio_path.write_bytes(await audio.read())

    mp4_path = generate_talking_head(
        face_image_path=str(face_path),
        audio_path=str(audio_path),
        output_root=str(workdir),
        still=True,
        pose_style=0,
    )

    vid_b64 = base64.b64encode(Path(mp4_path).read_bytes()).decode("utf-8")

    return {
        "ok": True,
        "filename": Path(mp4_path).name,
        "video_b64": vid_b64,
    }
