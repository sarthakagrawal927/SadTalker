import uuid
import subprocess
import sys
from pathlib import Path


def generate_talking_head(
    face_image_path: str,
    audio_path: str,
    output_root: str,
    still: bool = True,
    pose_style: int = 0,
):
    job_id = str(uuid.uuid4())
    job_dir = Path(output_root) / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    result_dir = job_dir / "results"
    result_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,  # <-- THIS is critical
        "inference.py",
        "--driven_audio",
        audio_path,
        "--source_image",
        face_image_path,
        "--result_dir",
        str(result_dir),
        "--pose_style",
        str(pose_style),
    ]
    if still:
        cmd.append("--still")

    repo_root = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        cmd,
        cwd=str(repo_root),
        check=False,
        capture_output=True,
        text=True,
    )

    if completed.returncode != 0:
        print("SadTalker stdout:\n", completed.stdout)
        print("SadTalker stderr:\n", completed.stderr)
        raise RuntimeError(f"SadTalker failed with code {completed.returncode}")

    mp4_path = None
    for f in result_dir.iterdir():
        if f.suffix.lower() == ".mp4":
            mp4_path = f
            break

    if mp4_path is None:
        raise RuntimeError("SadTalker produced no mp4 output")

    return str(mp4_path)
