import modal

# This image is the container template for your service.
image = (
    modal.Image.debian_slim(python_version="3.9")
    # ffmpeg is needed by SadTalker to mux audio/video
    .apt_install("ffmpeg")
    # install the bulk of your deps
    .pip_install_from_requirements("requirements.txt")
    # (optional) if you need CUDA torch for GPU inference, see note below
    # .pip_install(
    #     "torch",
    #     "torchvision",
    #     "torchaudio",
    #     index_url="https://download.pytorch.org/whl/cu121",
    # )
    # put your whole repo into /workspace inside the container
    # add_local_dir(...) is the new API replacing copy_local_dir(...)
    # default copy=False means "mount the dir at runtime", which is fine for us
    .add_local_dir(
        local_path=".",  # your SadTalker repo root on your machine
        remote_path="/workspace",
        copy=False,  # you can flip to True later to bake into the image
    )
)

app = modal.App("sadtalker-avatar-service")


@app.function(
    image=image,
    gpu="A10G",  # A10G/whatever Modal gives you. This spins up GPU on request.
    timeout=600,  # seconds per request
    secrets=[modal.Secret.from_name("custom-secret")],
)
@modal.asgi_app()
def api():
    import os, sys

    # match local behavior so inference.py finds ./checkpoints, etc.
    os.chdir("/workspace")
    sys.path.insert(0, "/workspace")

    # now we can import like we do locally
    from service.server import app as fastapi_app

    return fastapi_app
