````markdown
# SadTalker on macOS (Apple Silicon) – Survival Guide

This doc is for Future You (and any unlucky soul on an M1/M2 Mac) who wants to run SadTalker locally and generate “AI avatar talking my script” video.

It covers:
1. Clean install steps on macOS (conda, Python 3.8)
2. All the errors we hit and how we fixed them
3. Patches/hacks we had to make
4. What still sucks / what to offload to GPU later

This should live as `SADTALKER_MAC_SETUP.md` in your repo.


---

## 0. TL;DR

SadTalker was built for:
- Linux
- NVIDIA CUDA GPUs
- Old PyTorch (1.12.x)
- Old Python (3.8/3.9)

You are on:
- macOS Apple Silicon (arm64)
- No CUDA/GPU that SadTalker expects
- Newer libs with breaking changes

So you have to:
- Use conda + Python 3.8
- Manually install specific libs (torch, opencv, etc.)
- Patch some source files
- Bypass GFPGAN enhancer
- Download model checkpoints
- Chunk audio later for production

Once done: it does work locally. It will generate a talking-head MP4 from your voice + face image.

For production, you are 100% going to ship this in Docker on a rented GPU box. Don’t run inference for users on your Mac long-term. More on that later.


---

## 1. Prereqs / Assumptions

You need:
- macOS on Apple Silicon (M1/M2/etc.)
- Homebrew
- Conda (Miniforge / Mambaforge is ideal for arm64)
- `ffmpeg` via brew
- Git
- A face image (frontal, clear, PNG/JPG)
- An audio clip (.wav ideally, but .mp3 can work)

We’ll assume:
- Project folder: `~/SadTalker`
- Conda env name: `sadtalker38`
- Python version: 3.8

Yes, Python 3.8. Not 3.12. Don’t fight this.


---

## 2. Clone + Conda Setup

### Step 1: Fresh clone
```bash
cd ~
rm -rf SadTalker
git clone https://github.com/OpenTalker/SadTalker.git
cd SadTalker
````

### Step 2: Initialize conda (only needs to be done once on your machine)

```bash
conda init zsh
# then CLOSE TERMINAL and open a new one
```

If `conda init zsh` says `conda: command not found`, source your miniforge install first (e.g. `source ~/miniforge3/bin/activate`) and then run `conda init zsh`. Then open a fresh terminal.

### Step 3: Create + activate env

```bash
conda create -n sadtalker38 python=3.8 -y
conda activate sadtalker38
```

Check you’re actually in it:

```bash
echo $CONDA_DEFAULT_ENV        # should be sadtalker38
python --version               # should be 3.8.x
which python                   # should point into .../envs/sadtalker38/bin/python
```

---

## 3. Core Dependencies

### 3.1 Install PyTorch

SadTalker expects torch ~1.12.1 (CUDA-era), which doesn’t always have clean Mac arm wheels.
We try 1.12 first, fall back to 2.1.x if needed.

Try:

```bash
python -m pip install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1
```

If pip can’t find wheels for arm64:

```bash
python -m pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0
```

Sanity check:

```bash
python - << 'EOF'
import torch
print("torch ok:", torch.__version__)
try:
    print("mps available:", torch.backends.mps.is_available())
except:
    print("no mps attr but torch imported")
EOF
```

If this prints a version, torch is alive. You’re good.

### 3.2 ffmpeg (host-level)

```bash
brew install ffmpeg
ffmpeg -version
```

### 3.3 Install requirements.txt (first pass)

Now that torch exists, install the rest:

```bash
python -m pip install -r requirements.txt
```

This will partially work and partially fail. We’ll mop up manually.

---

## 4. Manual Installs / Fixes for Missing Modules

SadTalker imports a bunch of stuff. macOS arm + Py3.8 means some of it won’t install cleanly by default. Below is the hit list and fixes.

You will run inference, get an error, fix it, rerun. This is the order we actually hit them:

### 4.1 `No module named 'cv2'`

You’re missing OpenCV.

Fix:

```bash
python -m pip install opencv-python
# if that fails, try:
# python -m pip install opencv-python-headless
```

Sanity:

```bash
python - << 'EOF'
import cv2
print("cv2 OK:", cv2.__version__)
EOF
```

### 4.2 `No module named 'tqdm'`

Progress bar lib.

```bash
python -m pip install tqdm
```

### 4.3 `No module named 'skimage'`

This is actually `scikit-image`. On M1 + Py3.8, pip will try to compile it with Cython and you’ll cry. Use conda-forge instead:

```bash
conda install -c conda-forge scikit-image=0.19.3 -y
```

Sanity:

```bash
python - << 'EOF'
import skimage
from skimage import transform
print("skimage OK:", skimage.__version__)
EOF
```

### 4.4 `No module named 'kornia'`

Computer vision ops.

```bash
python -m pip install kornia==0.6.9
```

### 4.5 `No module named 'face_alignment'`

Facial landmark detector.

```bash
python -m pip install face-alignment==1.3.5
```

### 4.6 `No module named 'facexlib'`

Face detection / alignment utils SadTalker relies on.

```bash
python -m pip install facexlib==0.3.0
```

### 4.7 `charset_normalizer` blowing up deep inside `requests`

Error looked like:

```text
AttributeError: partially initialized module 'charset_normalizer' ...
```

Cause: new `charset-normalizer` ships compiled mypyc bits that don’t like Py3.8 + arm combo.

Fix:

```bash
python -m pip uninstall -y charset-normalizer
python -m pip install "charset-normalizer==3.3.2"
```

Sanity:

```bash
python - << 'EOF'
import requests, charset_normalizer
print("requests OK")
print("charset_normalizer OK", charset_normalizer.__version__)
EOF
```

### 4.8 `No module named 'yacs'`

Config helper from Facebook.

```bash
python -m pip install yacs==0.1.8
```

### 4.9 `No module named 'librosa'`

Audio loader/feature extractor. Installing via pip on M1 with Py3.8 triggers numba/llvmlite compile hell. Use conda-forge:

```bash
conda install -c conda-forge librosa=0.9.2 -y
```

Sanity:

```bash
python - << 'EOF'
import librosa
print("librosa OK:", librosa.__version__)
EOF
```

---

## 5. Two Nasty Ones We Had To Patch

### 5.1 The GFPGAN / basicsr mess

SadTalker tries to use GFPGAN/RealESRGAN to “enhance” the face. Those depend on `basicsr`. `basicsr` fails to build cleanly on macOS arm64 + Py3.8. We do NOT actually need this enhancer to prove the concept.

So we stubbed it out.

File: `src/utils/face_enhancer.py`

Replace the entire file with this:

```python
# dummy face_enhancer for Mac local inference without GFPGAN / basicsr

import cv2
import numpy as np

def enhancer_list():
    # no enhancers available in this build
    return []

def enhancer_generator_with_len(args):
    def _passthrough(img_list):
        # return frames untouched + fake lens info
        lens = []
        for img in img_list:
            if hasattr(img, "shape"):
                lens.append(img.shape[0])
            else:
                lens.append(0)
        return img_list, lens
    return _passthrough, []
```

Result:

* SadTalker will stop importing `gfpgan` and `basicsr`.
* You don’t need `--enhancer gfpgan` in your inference call.
* Output might be a bit less “beautified,” but it works. This is enough for MVP.

### 5.2 NumPy broke `np.float`

Newer NumPy removed `np.float`. Old code still calls `np.float`.

Error looked like:

```text
AttributeError: module 'numpy' has no attribute 'float'
```

Fix: add a shim at the top of `src/face3d/util/my_awing_arch.py` (right after `import numpy as np`):

```python
# compatibility shim for newer numpy
import numpy as np
if not hasattr(np, "float"):
    np.float = float
```

Now `preds.astype(np.float)` works again.

### 5.3 Shape bug in `align_img`

Error:

```text
ValueError: setting an array element with a sequence...
trans_params = np.array([w0, h0, s, t[0], t[1]])
```

Cause:

* `t[0]` / `t[1]` sometimes aren’t scalars, they’re 1-element arrays. NumPy refuses to make a flat vector.

Fix in `src/face3d/util/preprocess.py`, inside `align_img`, replace the line that builds `trans_params` with this safer block:

```python
    # force everything to scalar floats
    w0 = float(np.array(w0).reshape(-1)[0])
    h0 = float(np.array(h0).reshape(-1)[0])
    s  = float(np.array(s).reshape(-1)[0])

    t = np.array(t).reshape(-1)
    tx = float(t[0])
    ty = float(t[1])

    trans_params = np.array([w0, h0, s, tx, ty], dtype=np.float32)
```

Now it won’t choke on array shapes during alignment.

---

## 6. Downloading Checkpoints (If You Don’t, It Will Crash)

Even after all deps install, SadTalker will die like this:

```text
FileNotFoundError: [Errno 2] No such file or directory: './checkpoints/epoch_20.pth'
```

That’s expected. The repo doesn’t ship model weights.

You must:

1. Create a `checkpoints/` folder in the SadTalker root.
2. Download all the pretrained `.pth` / `.ckpt` files the README points to (sometimes via `bash scripts/download_models.sh`, sometimes via Google Drive/HF links).

   * This includes things like `epoch_20.pth` (3DMM recon), audio-to-motion models, expression/motion mapping models, renderer weights, etc.
3. Put them all in `./checkpoints/`.

After that, rerun inference. No checkpoints = no animation. This is normal.

---

## 7. Running Inference

You’ll need:

* A face image (frontal headshot). Call it `face.png`.
* An audio file. Better if you convert your .mp3 to mono 16kHz .wav for stability:

  ```bash
  ffmpeg -i sample.mp3 -ac 1 -ar 16000 audio.wav
  ```

Then run from inside the SadTalker repo:

```bash
conda activate sadtalker38

python inference.py \
  --driven_audio ./audio.wav \
  --source_image ./face.png \
  --result_dir ./results \
  --still \
  --pose_style 0
```

Notes:

* We intentionally do **not** pass `--enhancer gfpgan`, because we stubbed enhancer out to avoid basicsr/gfpgan on M1.
* `--still` tries to keep head motion subtle.
* `--pose_style 0` keeps motion calm. You can play with higher numbers later for more head bobbing.

If everything is good, you’ll see logs like:

* `3DMM Extraction for source image`
* `3DMM Extraction In Video:: 0%|...`
* tqdm bars
* it’ll spit an MP4 in `./results/`

Open that MP4. That’s your talking avatar.

If it fails on face landmarks / “no face found,” try a clearer, front-facing human portrait (not profile, not sunglasses, not tiny face).

---

## 8. Performance and Audio Length

* This runs on CPU/MPS on Mac, so it’s slow. Frame count = render time. A 20s clip will feel like forever.
* SadTalker quality degrades for very long, continuous takes (face starts looking robotic). The practical sweet spot is ~10–20 seconds per chunk.
* Strategy:

  * Split your script into short sentences (10–15s of speech each).
  * Generate each chunk separately.
  * Concatenate the resulting MP4s with ffmpeg.
  * Add captions/b-roll in post.

Later you can automate all of that and call it a “reel generator.”

---

## 9. Why Production Should Not Run On Your Mac

Local Mac setup is just proof-of-concept. Real deployment is easier than what you just did:

### On a rented GPU box (RunPod / Vast / etc.):

* Start with an Ubuntu + CUDA base image
* Use Python 3.8 or 3.9
* `pip install` torch==1.12.1+cu113, torchvision==0.13.1+cu113, etc.
* Install requirements.txt (now it actually works because this is literally what SadTalker expects: Linux x86_64 + CUDA)
* Keep the original `face_enhancer.py` (with GFPGAN) because basicsr will build fine there
* Bundle checkpoints into the Docker image
* Write a tiny FastAPI/Flask `serve.py` that:

  * accepts an image + audio
  * runs inference
  * returns the mp4 path / binary
* Done. That’s your “/generateAvatarVideo” API.

Scaling = spin more GPU pods.

Buying a 4090 rig only makes sense once you’re hammering GPUs 24/7. Until then, renting at ~$0.2–$0.35/hr (ballpark for 4090-class on GPU marketplaces) is cheaper than owning.

---

## 10. Mental Model / What You Built

You now have a working pipeline that can become a product:

1. **TTS / Voice Generation**

   * Take script → generate expressive speech (Fish / OpenAudio / etc.) → `audio.wav`

2. **SadTalker Avatar**

   * Take `audio.wav` + `face.png` → generate talking-head video clip

3. **Post-processing**

   * Stitch multiple clips
   * Add captions, b-roll, CTA overlays
   * Output vertical 9:16 reel

Step 2 was the nasty part. You just got it working locally on an M1, which is honestly not supposed to be pleasant.

Now you can automate steps 1–3 and sell “paste script → get human talking video,” then later migrate inference to GPU infra.

---

## 11. Quick Troubleshooting Cheat Sheet

**Problem:** `conda activate sadtalker38` says “Run 'conda init' first.”
**Fix:**

```bash
conda init zsh
# close terminal and reopen
```

---

**Problem:** `ImportError: No module named torch` when running `inference.py`
**Fix:**

1. Make sure `conda activate sadtalker38` is active.
2. Install torch inside that env using `python -m pip install ...` (not `pip` alone).
3. Confirm with:

   ```bash
   python -c "import torch; print(torch.__version__)"
   ```

---

**Problem:** `ModuleNotFoundError: cv2/tqdm/skimage/kornia/facexlib/face_alignment/yacs/librosa`
**Fix:** see section 4 and install them one by one.
Use `conda install` for heavy scientific libs (scikit-image, librosa), `pip install` for light pure-Python stuff (yacs, tqdm), `pip install facexlib==0.3.0`, etc.

---

**Problem:** `charset_normalizer` / `requests` causes AttributeError about `md__mypyc`
**Fix:**

```bash
python -m pip uninstall -y charset-normalizer
python -m pip install "charset-normalizer==3.3.2"
```

---

**Problem:** `ModuleNotFoundError: gfpgan` / basicsr build fails
**Fix:** we bypass enhancer completely. Replace `src/utils/face_enhancer.py` with the stub in section 5.1.
Also, don’t pass `--enhancer gfpgan` in inference.

---

**Problem:** `AttributeError: numpy has no attribute 'float'`
**Fix:** add this to `src/face3d/util/my_awing_arch.py`:

```python
import numpy as np
if not hasattr(np, "float"):
    np.float = float
```

---

**Problem:** `ValueError: setting an array element with a sequence` from `align_img`
**Fix:** patch `src/face3d/util/preprocess.py` to coerce scalars before building `trans_params`, as in section 5.3.

---

**Problem:** `FileNotFoundError: ./checkpoints/epoch_20.pth`
**Fix:** you haven’t downloaded the model weights. Create `checkpoints/`, download all the pretrained .pth/.ckpt files that SadTalker expects (the repo usually gives links or a download script), drop them there, rerun.

---

**Problem:** Runtime says it can’t detect landmarks / can’t find face
**Fix:** try a clean, front-facing headshot with the face big and visible. Side profiles, sunglasses, or tiny cropped faces often fail.

---

**Problem:** It feels insanely slow / “it’ll take an hour XD”
**Fix:** welcome to CPU/MPS inference on an M1 doing frame-by-frame animation.
This is normal. For production, you will not use your Mac. You’ll send requests to a GPU pod via API.

---

## 12. Final Notes

* You do NOT need GFPGAN/basicsr locally to prove this works.

* You do NOT need perfect realtime performance locally.

* The only goal of the Mac setup is: “Can I generate an AI talking head from any script+voice I control?”
  You can now: yes.

* From here:

  * Wrap SadTalker inference in a python API (`/generate`).
  * Move that whole thing into a Docker image on a rented 4090.
  * Add TTS (Fish etc) before it.
  * Add stitching / captions after it.
  * You’ve got a vertical video generator backend.

Ship it.

```

::contentReference[oaicite:0]{index=0}
```
