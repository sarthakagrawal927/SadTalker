import modal

image = (
    modal.Image.debian_slim(python_version="3.9")
    # ffmpeg is required for SadTalker’s video+audio mux step
    .apt_install("ffmpeg")
    # install everything you used locally (torch, facexlib, fastapi, uvicorn, imageio-ffmpeg, etc.)
    .pip_install_from_requirements("requirements.txt")
    # copy ALL your code + checkpoints into the container FS at /workspace
    .copy_local_dir(".", "/workspace")
)

app = modal.App("sadtalker-avatar-service")


@app.function(
    image=image,
    gpu="A10G",  # A10G is good, T4/L4 etc also fine
    timeout=600,  # 10 min per request window, bump if you do long clips
)
@modal.asgi_app()
def api():
    import sys, os

    # make container behave like your local repo root
    os.chdir("/workspace")
    sys.path.insert(0, "/workspace")

    # now import your FastAPI app
    from service.server import app as fastapi_app

    return fastapi_app
