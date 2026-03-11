

import os
import json
import shutil
import zipfile
from pathlib import Path
from google.colab import files

ROOT_OUT = Path(OUTPUTS_DIR)
SUBMIT_DIR = ROOT_OUT / "submission_package"
ZIP_PATH = ROOT_OUT / f"musicgen_hw4_submission_{RUN_TAG}.zip"

EXPORT_DIR = ROOT_OUT / "exported_musicgen_ft"
GEN_DIR = ROOT_OUT / "generated_prompts"


if SUBMIT_DIR.exists():
    shutil.rmtree(SUBMIT_DIR)
SUBMIT_DIR.mkdir(parents=True, exist_ok=True)

if ZIP_PATH.exists():
    ZIP_PATH.unlink()


weights_dir = SUBMIT_DIR / "weights_exported"
weights_dir.mkdir(parents=True, exist_ok=True)

for name in ["state_dict.bin", "compression_state_dict.bin"]:
    src = EXPORT_DIR / name
    if src.exists():
        shutil.copy2(src, weights_dir / name)
    else:
        print(f"[warn] not found: {src}")


gen_submit_dir = SUBMIT_DIR / "generated_wavs"
gen_submit_dir.mkdir(parents=True, exist_ok=True)

for i in range(1, 6):
    src = GEN_DIR / f"prompt_{i}.wav"
    if src.exists():
        shutil.copy2(src, gen_submit_dir / src.name)
    else:
        print(f"[warn] not found: {src}")


extra_files = [
    GEN_DIR / "test_prompts.json",
    GEN_DIR / "generation_stdout.log",
    Path(TRAIN_LOG_PATH),
    Path(INSTALL_LOG_PATH),
]

for src in extra_files:
    if src.exists():
        shutil.copy2(src, SUBMIT_DIR / src.name)
    else:
        print(f"[warn] optional file not found: {src}")


report = {
    "run_tag": RUN_TAG,
    "outputs_dir": str(ROOT_OUT),
    "latest_xp_dir": str(latest_xp_dir) if latest_xp_dir is not None else None,
    "checkpoint_path": str(checkpoint_path) if "checkpoint_path" in globals() else None,
    "export_dir": str(EXPORT_DIR),
    "generated_dir": str(GEN_DIR),
    "train_epochs": globals().get("TRAIN_EPOCHS", None),
    "updates_per_epoch": globals().get("UPDATES_PER_EPOCH", None),
    "learning_rate": globals().get("LEARNING_RATE", None),
    "generated_files": [f"prompt_{i}.wav" for i in range(1, 6)],
}

with open(SUBMIT_DIR / "report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)


with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
    for path in SUBMIT_DIR.rglob("*"):
        if path.is_file():
            zf.write(path, arcname=path.relative_to(SUBMIT_DIR))

print("✓ submission dir:", SUBMIT_DIR)
print("✓ zip created:", ZIP_PATH)


files.download(str(ZIP_PATH))
