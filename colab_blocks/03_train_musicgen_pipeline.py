

import os
import sys
import gc
import re
import json
import gzip
import wave
import random
import shutil
import signal
import subprocess
import getpass
import time
from pathlib import Path
from tqdm import tqdm


subprocess.run([sys.executable, "-m", "pip", "install", "-q", "wandb", "requests"], check=True)

import wandb

LOCAL_DATA_DIR = "/content/musiccaps_data"
REPO_CLONE_DIR = "/content/te2_repo"

RUN_TAG = time.strftime("%Y%m%d_%H%M%S")

MANIFESTS_DIR = "/content/manifests"
AUDIOCRAFT_DIR = "/content/audiocraft"
EGS_DIR = "/content/egs/musiccaps"


OUTPUTS_DIR = f"/content/outputs_{RUN_TAG}"

TRAIN_LOG_PATH = os.path.join(OUTPUTS_DIR, "train_stdout.log")
INSTALL_LOG_PATH = os.path.join(OUTPUTS_DIR, "audiocraft_install.log")
DATASET_CARD_PATH = os.path.join(OUTPUTS_DIR, "dataset_card.md")

GITHUB_USER = "dgokr2-wq"
GITHUB_REPO = "te2"
GITHUB_BRANCH = "main"

WANDB_PROJECT = "musicgen-musiccaps-ft"
WANDB_GROUP = "colab-finetune"
WANDB_SETUP_RUN_NAME = f"musiccaps-setup-{RUN_TAG}"
WANDB_TRAIN_RUN_NAME = f"musicgen-small-long-{RUN_TAG}"
WANDB_POST_RUN_NAME = f"musicgen-postprocess-{RUN_TAG}"

MICROMAMBA_ROOT = "/content/micromamba_root"
ENV_NAME = "audiocraft310"


TRAIN_EPOCHS = 50
UPDATES_PER_EPOCH = 150
LEARNING_RATE = 8e-6

os.makedirs(MANIFESTS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(f"{EGS_DIR}/train", exist_ok=True)
os.makedirs(f"{EGS_DIR}/valid", exist_ok=True)


def run(cmd, cwd=None, env=None, print_cmd=True, secret=False, capture_output=False, check=True):
    if print_cmd:
        if secret:
            print("> [secret command hidden]")
        else:
            print(">", " ".join(cmd))
    return subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        check=check,
        capture_output=capture_output
    )

def run_live(cmd, cwd=None, env=None, print_cmd=True, secret=False, log_path=None):
    if print_cmd:
        if secret:
            print("> [secret command hidden]")
        else:
            print(">", " ".join(cmd))

    log_f = open(log_path, "w", encoding="utf-8") if log_path else None
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    try:
        for line in iter(proc.stdout.readline, ""):
            print(line, end="")
            if log_f:
                log_f.write(line)
        proc.wait()
    finally:
        if log_f:
            log_f.close()
    if proc.returncode != 0:
        raise RuntimeError(f"Команда упала с кодом {proc.returncode}")

def normalize_string(x):
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    return str(x).strip()

def normalize_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return [normalize_string(v) for v in x if normalize_string(v)]
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        if "," in s:
            return [p.strip() for p in s.split(",") if p.strip()]
        return [s]
    return [str(x).strip()]

def normalize_music_json(meta):
    norm = {
        "description": normalize_string(meta.get("description", "")),
        "general_mood": normalize_string(meta.get("general_mood", "")),
        "genre_tags": normalize_list(meta.get("genre_tags", [])),
        "lead_instrument": normalize_string(meta.get("lead_instrument", "")),
        "accompaniment": normalize_string(meta.get("accompaniment", "")),
        "tempo_and_rhythm": normalize_string(meta.get("tempo_and_rhythm", "")),
        "vocal_presence": normalize_string(meta.get("vocal_presence", "")),
        "production_quality": normalize_string(meta.get("production_quality", "")),
    }
    return norm


DATA_DIR = None

if os.path.isdir(LOCAL_DATA_DIR):
    DATA_DIR = LOCAL_DATA_DIR
    print(f"✓ Использую локальные данные: {DATA_DIR}")
else:
    print("Локальная папка /content/musiccaps_data не найдена.")
    print("Пробую скачать датасет из GitHub...")

    run(["apt-get", "update", "-qq"])
    run(["apt-get", "install", "-y", "git-lfs"])
    run(["git", "lfs", "install"])

    if os.path.exists(REPO_CLONE_DIR):
        shutil.rmtree(REPO_CLONE_DIR)

    gh_token = getpass.getpass(
        "Если репозиторий private — вставь GitHub token. "
        "Если public — просто нажми Enter: "
    ).strip()

    if gh_token:
        repo_url = f"https://{GITHUB_USER}:{gh_token}@github.com/{GITHUB_USER}/{GITHUB_REPO}.git"
        secret_mode = True
    else:
        repo_url = f"https://github.com/{GITHUB_USER}/{GITHUB_REPO}.git"
        secret_mode = False

    run(["git", "clone", "--branch", GITHUB_BRANCH, repo_url, REPO_CLONE_DIR], secret=secret_mode)
    run(["git", "lfs", "pull"], cwd=REPO_CLONE_DIR)

    candidate_dirs = [
        os.path.join(REPO_CLONE_DIR, "musiccaps_data"),
        os.path.join(REPO_CLONE_DIR, "data"),
        REPO_CLONE_DIR,
    ]

    for cand in candidate_dirs:
        if os.path.isdir(cand):
            try:
                wav_count = len([f for f in os.listdir(cand) if f.endswith(".wav")])
                json_count = len([f for f in os.listdir(cand) if f.endswith(".json")])
            except Exception:
                wav_count, json_count = 0, 0

            if wav_count > 0 or json_count > 0:
                DATA_DIR = cand
                break

    if DATA_DIR is None:
        raise RuntimeError(
            f"Не нашёл папку с wav/json внутри репозитория {GITHUB_USER}/{GITHUB_REPO}. "
            f"Ожидалась папка musiccaps_data в корне."
        )

    print(f"✓ Данные подтянуты из GitHub: {DATA_DIR}")

assert DATA_DIR is not None and os.path.isdir(DATA_DIR), f"Не найдена папка с данными: {DATA_DIR}"


WANDB_API_KEY = getpass.getpass("Вставь W&B API key: ").strip()
assert WANDB_API_KEY, "W&B API key пустой"

os.environ["WANDB_API_KEY"] = WANDB_API_KEY
os.environ["WANDB_PROJECT"] = WANDB_PROJECT
os.environ["WANDB_DIR"] = OUTPUTS_DIR
os.environ["WANDB_SILENT"] = "true"

print("Проверяем W&B ключ...")
ok = wandb.login(key=WANDB_API_KEY, relogin=True, verify=True)
assert ok, "W&B login не прошел"

api = wandb.Api()
try:
    viewer = api.viewer
    print(f"✓ W&B login ok. Пользователь: {getattr(viewer, 'username', 'unknown')}")
except Exception:
    print("✓ W&B login ok")


print("\n===== CLEANUP BEFORE TRAIN =====")

for var_name in [
    "model", "tokenizer", "generator", "pipe",
    "dataset", "outputs", "quantization_config"
]:
    if var_name in globals():
        try:
            del globals()[var_name]
        except:
            pass

try:
    out = subprocess.check_output(["ps", "-eo", "pid,cmd"], text=True)
    my_pid = os.getpid()
    for line in out.splitlines()[1:]:
        parts = line.strip().split(None, 1)
        if len(parts) != 2:
            continue
        pid_str, cmdline = parts
        try:
            pid = int(pid_str)
        except:
            continue

        if pid == my_pid:
            continue

        bad = (
            "python -m dora run" in cmdline or
            "audiocraft/train.py" in cmdline or
            "accelerate launch" in cmdline or
            ("micromamba run -n" in cmdline and "dora run" in cmdline)
        )
        if bad:
            try:
                os.kill(pid, signal.SIGKILL)
                print(f"[cleanup] killed pid={pid}: {cmdline[:120]}")
            except Exception:
                pass
except Exception as e:
    print("[cleanup] process cleanup warning:", e)

gc.collect()

try:
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print("[cleanup] CUDA cache cleared")
except Exception as e:
    print("[cleanup] torch cleanup warning:", e)

print("\n===== GPU AFTER CLEANUP =====")
subprocess.run(["nvidia-smi"], check=False)


entries = []
files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".wav")])

print("\nСобираем manifests...")
for wav_file in tqdm(files, desc="Manifest"):
    base_name = os.path.splitext(wav_file)[0]
    wav_path = os.path.join(DATA_DIR, wav_file)
    json_path = os.path.join(DATA_DIR, f"{base_name}.json")

    if not os.path.exists(json_path):
        continue

    try:
        with wave.open(wav_path, "rb") as wav_f:
            frames = wav_f.getnframes()
            sample_rate = wav_f.getframerate()
            duration = frames / float(sample_rate)

        with open(json_path, "r", encoding="utf-8") as f:
            raw_meta = json.load(f)

        norm_meta = normalize_music_json(raw_meta)

        
        if raw_meta != norm_meta:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(norm_meta, f, ensure_ascii=False, indent=2)
                
        entry = {
            "path": wav_path,
            "duration": duration,
            "sample_rate": sample_rate,
            "amplitude": None,
            "weight": None,
            "info_path": None, 
            "description": norm_meta["description"],
            "general_mood": norm_meta["general_mood"],
            "genre_tags": norm_meta["genre_tags"],
            "lead_instrument": norm_meta["lead_instrument"],
            "accompaniment": norm_meta["accompaniment"],
            "tempo_and_rhythm": norm_meta["tempo_and_rhythm"],
            "vocal_presence": norm_meta["vocal_presence"],
            "production_quality": norm_meta["production_quality"],
        }
        entries.append(entry)

    except Exception as e:
        print(f"[skip] {base_name}: {type(e).__name__} - {e}")

num_valid = len(entries)
print(f"Валидных треков: {num_valid}")
if num_valid == 0:
    raise RuntimeError(f"Не найдено ни одного валидного WAV+JSON pair в {DATA_DIR}")

random.seed(42)
random.shuffle(entries)

split_idx = max(1, int(num_valid * 0.95))
train_entries = entries[:split_idx]
valid_entries = entries[split_idx:] if len(entries[split_idx:]) > 0 else entries[:1]

def write_jsonl_gz(data, filename):
    with gzip.open(filename, "wt", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

train_manifest = os.path.join(MANIFESTS_DIR, "train.jsonl.gz")
valid_manifest = os.path.join(MANIFESTS_DIR, "valid.jsonl.gz")

write_jsonl_gz(train_entries, train_manifest)
write_jsonl_gz(valid_entries, valid_manifest)

shutil.copy(train_manifest, f"{EGS_DIR}/train/data.jsonl.gz")
shutil.copy(valid_manifest, f"{EGS_DIR}/valid/data.jsonl.gz")

print("✓ manifests готовы")
print("  ", train_manifest)
print("  ", valid_manifest)


dataset_card = f"""# MusicCaps fine-tuning dataset

- total_valid_pairs: {num_valid}
- train_size: {len(train_entries)}
- valid_size: {len(valid_entries)}
- audio_dir: {DATA_DIR}
- manifests_dir: {MANIFESTS_DIR}
- sample_rate: 32000
- channels: 1

This dataset contains:
- WAV files downloaded from MusicCaps YouTube references
- JSON sidecar metadata with fields:
  - description
  - general_mood
  - genre_tags
  - lead_instrument
  - accompaniment
  - tempo_and_rhythm
  - vocal_presence
  - production_quality

Manifest entries also contain:
- info_path
- duplicated structured metadata fields
"""

with open(DATASET_CARD_PATH, "w", encoding="utf-8") as f:
    f.write(dataset_card)


preview_table = wandb.Table(columns=[
    "ytid", "audio", "duration_sec", "description",
    "general_mood", "genre_tags", "lead_instrument"
])

preview_candidates = []
for wav_file in files:
    base_name = os.path.splitext(wav_file)[0]
    wav_path = os.path.join(DATA_DIR, wav_file)
    json_path = os.path.join(DATA_DIR, f"{base_name}.json")
    if os.path.exists(json_path):
        preview_candidates.append((base_name, wav_path, json_path))
    if len(preview_candidates) >= 8:
        break

for base_name, wav_path, json_path in preview_candidates:
    try:
        with wave.open(wav_path, "rb") as wav_f:
            duration = wav_f.getnframes() / float(wav_f.getframerate())
        with open(json_path, "r", encoding="utf-8") as f:
            meta = normalize_music_json(json.load(f))

        preview_table.add_data(
            base_name,
            wandb.Audio(wav_path, caption=meta.get("description", "")),
            round(duration, 3),
            meta.get("description", ""),
            meta.get("general_mood", ""),
            ", ".join(meta.get("genre_tags", [])),
            meta.get("lead_instrument", "")
        )
    except Exception as e:
        print(f"[preview skip] {base_name}: {e}")


setup_config = {
    "num_valid_pairs": num_valid,
    "train_size": len(train_entries),
    "valid_size": len(valid_entries),
    "audio_dir": DATA_DIR,
    "manifests_dir": MANIFESTS_DIR,
    "sample_rate": 32000,
    "channels": 1,
    "segment_duration_train": 2,
    "model_scale": "small",
    "source_repo": f"{GITHUB_USER}/{GITHUB_REPO}",
    "env_python": "3.10 (micromamba)",
}

with wandb.init(
    project=WANDB_PROJECT,
    group=WANDB_GROUP,
    name=WANDB_SETUP_RUN_NAME,
    job_type="setup",
    config=setup_config,
) as setup_run:
    setup_run.log({"dataset/preview_table": preview_table})

    manifest_artifact = wandb.Artifact(
        name=f"musiccaps-manifests-{setup_run.id}",
        type="dataset",
        description="Train/valid manifests and dataset card for MusicCaps fine-tuning.",
        metadata={
            "train_size": len(train_entries),
            "valid_size": len(valid_entries),
            "num_valid_pairs": num_valid,
        },
    )
    manifest_artifact.add_file(train_manifest)
    manifest_artifact.add_file(valid_manifest)
    manifest_artifact.add_file(DATASET_CARD_PATH)
    setup_run.log_artifact(manifest_artifact, aliases=["latest", "manifests"])

    dataset_artifact = wandb.Artifact(
        name=f"musiccaps-data-{setup_run.id}",
        type="dataset",
        description="Full local dataset directory containing WAV files and structured JSON metadata.",
        metadata={
            "num_valid_pairs": num_valid,
            "contains_wav": True,
            "contains_json": True,
        },
    )
    dataset_artifact.add_dir(DATA_DIR, name="musiccaps_data")
    setup_run.log_artifact(dataset_artifact, aliases=["latest", "full-data"])

print("✓ W&B setup artifacts logged")



print("\nСоздаём новое env под Blackwell / CUDA 12.8...")


ENV_NAME = "audiocraft_bw"
MICROMAMBA_ROOT = "/content/micromamba_root"

MAMBA_ENV = os.environ.copy()
MAMBA_ENV["MAMBA_ROOT_PREFIX"] = MICROMAMBA_ROOT
os.makedirs(MICROMAMBA_ROOT, exist_ok=True)

MAMBA_BIN_CANDIDATES = [
    f"{MICROMAMBA_ROOT}/bin/micromamba",
    "/root/.local/bin/micromamba",
    "/root/bin/micromamba",
]

MAMBA_BIN = None
for cand in MAMBA_BIN_CANDIDATES:
    if os.path.exists(cand):
        MAMBA_BIN = cand
        break

if MAMBA_BIN is None:
    run_live(
        ["bash", "-lc", "curl -Ls https://micro.mamba.pm/install.sh | bash"],
        env=MAMBA_ENV
    )
    for cand in MAMBA_BIN_CANDIDATES:
        if os.path.exists(cand):
            MAMBA_BIN = cand
            break

assert MAMBA_BIN is not None and os.path.exists(MAMBA_BIN), "Не удалось найти micromamba"

run_live([MAMBA_BIN, "--version"], env=MAMBA_ENV)

env_list = run([MAMBA_BIN, "env", "list"], env=MAMBA_ENV, capture_output=True).stdout
if ENV_NAME in env_list:
    print(f"Удаляю старый env {ENV_NAME} ...")
    run_live([MAMBA_BIN, "env", "remove", "-y", "-n", ENV_NAME], env=MAMBA_ENV)


run_live([
    MAMBA_BIN, "create", "-y", "-n", ENV_NAME,
    "-c", "conda-forge",
    "python=3.10",
    "ffmpeg",
    "av=11.0.0",
    "git",
], env=MAMBA_ENV)

run_live([MAMBA_BIN, "run", "-n", ENV_NAME, "python", "--version"], env=MAMBA_ENV)

ENV_PREFIX = f"{MICROMAMBA_ROOT}/envs/{ENV_NAME}"

RUNTIME_ENV = os.environ.copy()
RUNTIME_ENV["MAMBA_ROOT_PREFIX"] = MICROMAMBA_ROOT
RUNTIME_ENV["CONDA_PREFIX"] = ENV_PREFIX
RUNTIME_ENV["PATH"] = f"{ENV_PREFIX}/bin:" + RUNTIME_ENV.get("PATH", "")
RUNTIME_ENV["LD_LIBRARY_PATH"] = f"{ENV_PREFIX}/lib:{ENV_PREFIX}/lib64:" + RUNTIME_ENV.get("LD_LIBRARY_PATH", "")
RUNTIME_ENV["PYTHONNOUSERSITE"] = "1"

preload_libs = []
for cand in [
    f"{ENV_PREFIX}/lib/libstdc++.so.6",
    f"{ENV_PREFIX}/lib/libgcc_s.so.1",
]:
    if os.path.exists(cand):
        preload_libs.append(cand)
if preload_libs:
    RUNTIME_ENV["LD_PRELOAD"] = ":".join(preload_libs)


if not os.path.isdir(AUDIOCRAFT_DIR):
    print("\nКлонируем AudioCraft...")
    run(["git", "clone", "https://github.com/facebookresearch/audiocraft.git", AUDIOCRAFT_DIR])


music_dataset_path = Path(f"{AUDIOCRAFT_DIR}/audiocraft/data/music_dataset.py")
assert music_dataset_path.exists(), f"Не найден файл: {music_dataset_path}"

text = music_dataset_path.read_text(encoding="utf-8")

def patch_once(pattern, repl, src, desc):
    new_src, n = re.subn(pattern, repl, src, count=1, flags=re.MULTILINE | re.DOTALL)
    if n == 0:
        raise RuntimeError(f"Не удалось применить патч: {desc}")
    return new_src

if "general_mood: tp.Optional[str] = None" not in text:
    text = patch_once(
        r"(instrument:\s*tp\.Optional\[str\]\s*=\s*None\s*\n)",
        r"\1"
        r"    general_mood: tp.Optional[str] = None\n"
        r"    genre_tags: tp.Optional[list] = None\n"
        r"    lead_instrument: tp.Optional[str] = None\n"
        r"    accompaniment: tp.Optional[str] = None\n"
        r"    tempo_and_rhythm: tp.Optional[str] = None\n"
        r"    vocal_presence: tp.Optional[str] = None\n"
        r"    production_quality: tp.Optional[str] = None\n",
        text,
        "добавление новых полей в MusicInfo"
    )

if "['moods', 'keywords', 'genre_tags']" not in text:
    text = patch_once(
        r"elif attribute in \['moods', 'keywords'\]:\s*\n\s*preprocess_func = get_keyword_list",
        "elif attribute in ['moods', 'keywords', 'genre_tags']:\n            preprocess_func = get_keyword_list",
        text,
        "genre_tags как keyword_list"
    )

if "general_mood" not in text.split("attribute_getter", 1)[-1]:
    text = patch_once(
        r"elif attribute in \['title', 'artist', 'description'\]:\s*\n\s*preprocess_func = get_string",
        "elif attribute in ['title', 'artist', 'description', 'general_mood', 'lead_instrument', 'accompaniment', 'tempo_and_rhythm', 'vocal_presence', 'production_quality']:\n            preprocess_func = get_string",
        text,
        "новые строковые поля"
    )

if "lead_instrument" not in text.split("valid_field_name", 1)[-1]:
    text = patch_once(
        r"valid_field_name = field_name in \['key', 'bpm', 'genre', 'moods', 'instrument', 'keywords'\]",
        "valid_field_name = field_name in ['key', 'bpm', 'genre', 'moods', 'instrument', 'keywords', 'general_mood', 'genre_tags', 'lead_instrument', 'accompaniment', 'tempo_and_rhythm', 'vocal_presence', 'production_quality']",
        text,
        "добавление новых полей в augment_music_info_description"
    )

music_dataset_path.write_text(text, encoding="utf-8")
print(f"✓ music_dataset.py пропатчен: {music_dataset_path}")


print("\nСтавим Blackwell-совместимый стек внутрь env...")

run_live(
    [MAMBA_BIN, "run", "-n", ENV_NAME, "python", "-m", "pip", "install", "-U", "pip", "setuptools", "wheel"],
    env=RUNTIME_ENV
)


run_live(
    [
        MAMBA_BIN, "run", "-n", ENV_NAME, "python", "-m", "pip", "install",
        "torch==2.7.1",
        "torchvision==0.22.1",
        "torchaudio==2.7.1",
        "--index-url", "https://download.pytorch.org/whl/cu128"
    ],
    env=RUNTIME_ENV,
    log_path=INSTALL_LOG_PATH
)


run_live(
    [
        MAMBA_BIN, "run", "-n", ENV_NAME, "python", "-m", "pip", "install",
        "xformers==0.0.31",
    ],
    env=RUNTIME_ENV,
    log_path=INSTALL_LOG_PATH
)


run_live(
    [
        MAMBA_BIN, "run", "-n", ENV_NAME, "python", "-m", "pip", "install",
        "numpy==1.26.4",
        "wandb==0.25.0",
        "requests==2.32.5",
        "tensorboard==2.20.0",
        "huggingface_hub==0.23.4",
        "tokenizers==0.19.1",
        "transformers==4.41.2",
        "flashy==0.0.2",
        "hydra-core==1.3.2",
        "hydra_colorlog==1.2.0",
        "julius==0.2.7",
        "num2words==0.5.14",
        "sentencepiece==0.2.1",
        "spacy==3.7.6",
        "librosa==0.11.0",
        "soundfile==0.13.1",
        "torchmetrics==1.9.0",
        "encodec==0.1.1",
        "demucs==4.0.1",
        "pesq==0.0.4",
        "pystoi==0.4.1",
        "torchdiffeq==0.2.5",
    ],
    env=RUNTIME_ENV,
    log_path=INSTALL_LOG_PATH
)

run_live(
    [MAMBA_BIN, "run", "-n", ENV_NAME, "python", "-m", "pip", "install", "-e", AUDIOCRAFT_DIR, "--no-deps"],
    env=RUNTIME_ENV,
    log_path=INSTALL_LOG_PATH
)

print("\nПроверяем итоговый стек внутри env...")
run_live(
    [
        MAMBA_BIN, "run", "-n", ENV_NAME,
        "python", "-c",
        (
            "import torch, torchvision, torchaudio, xformers; "
            "print('torch =', torch.__version__); "
            "print('torch cuda =', torch.version.cuda); "
            "print('cuda available =', torch.cuda.is_available()); "
            "print('device =', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'); "
            "print('capability =', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None); "
            "x = torch.randn(4, device='cuda'); "
            "print('tensor on cuda ok =', x.device); "
            "print('torchvision =', torchvision.__version__); "
            "print('torchaudio =', torchaudio.__version__); "
            "print('xformers =', xformers.__version__); "
        )
    ],
    env=RUNTIME_ENV
)

print("✓ Blackwell-совместимый стек и AudioCraft готовы")


cfg_dir = f"{AUDIOCRAFT_DIR}/config/dset/audio"
os.makedirs(cfg_dir, exist_ok=True)

cfg_text = f"""# @package __global__

datasource:
  max_sample_rate: 32000
  max_channels: 1

  train: {EGS_DIR}/train
  valid: {EGS_DIR}/valid
  evaluate: {EGS_DIR}/valid
  generate: {EGS_DIR}/valid
"""

cfg_path = f"{cfg_dir}/musiccaps_ft.yaml"
with open(cfg_path, "w", encoding="utf-8") as f:
    f.write(cfg_text)

print(f"✓ dset-конфиг создан: {cfg_path}")


try:
    import wandb
    wandb.finish()
except Exception:
    pass

TRAIN_ENV = RUNTIME_ENV.copy()
TRAIN_ENV["USER"] = "root"
TRAIN_ENV["AUDIOCRAFT_DORA_DIR"] = OUTPUTS_DIR
TRAIN_ENV["PYTHONPATH"] = AUDIOCRAFT_DIR + (":" + TRAIN_ENV["PYTHONPATH"] if "PYTHONPATH" in TRAIN_ENV else "")
TRAIN_ENV["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


print("ENV_PREFIX =", ENV_PREFIX)
print("OUTPUTS_DIR =", OUTPUTS_DIR)


cmd = [
    MAMBA_BIN, "run", "-n", ENV_NAME,
    "python", "-m", "dora", "run",
    "solver=musicgen/musicgen_base_32khz",
    "dset=audio/musiccaps_ft",
    "model/lm/model_scale=small",
    "continue_from=//pretrained/facebook/musicgen-small",

    
    "logging.log_wandb=false", 
    "logging.log_tensorboard=true",
    "logging.log_updates=5",

    "deadlock.use=false",
    "autocast=true",
    "autocast_dtype=float16",
    "optim.ema.use=false",

   
    "dataset.batch_size=1",
    "dataset.num_workers=0",
    "dataset.segment_duration=2", 
    "+dataset.info_fields_required=false",
    "dataset.valid.num_samples=2",
    "dataset.evaluate.num_samples=0",
    "dataset.generate.num_samples=0",

    "++conditioner.text.merge_text_p=0.25",
    "++conditioner.text.drop_desc_p=0.5",
    "++conditioner.text.drop_other_p=0.5",
    "++conditioner.text.text_attributes=[genre_tags,general_mood,lead_instrument,accompaniment,tempo_and_rhythm,vocal_presence,production_quality]",

    f"optim.epochs={TRAIN_EPOCHS}",
    f"optim.updates_per_epoch={UPDATES_PER_EPOCH}",
    f"optim.lr={LEARNING_RATE}",
]

print("\n🚀 ЗАПУСК ОБУЧЕНИЯ (без внутреннего W&B)...")
run_live(cmd, cwd=AUDIOCRAFT_DIR, env=TRAIN_ENV, log_path=TRAIN_LOG_PATH)


xps_dir = Path(OUTPUTS_DIR) / "xps"
latest_xp_dir = None

if xps_dir.exists():
    xp_dirs = [p for p in xps_dir.iterdir() if p.is_dir()]
    if xp_dirs:
        latest_xp_dir = max(xp_dirs, key=lambda p: p.stat().st_mtime)

print("\n✓ TRAIN FINISHED")
if latest_xp_dir is not None:
    print(f"✓ Latest XP dir: {latest_xp_dir}")


with wandb.init(
    project=WANDB_PROJECT,
    group=WANDB_GROUP,
    name=WANDB_POST_RUN_NAME,
    job_type="postprocess",
    config={
        "latest_xp_dir": str(latest_xp_dir) if latest_xp_dir else None,
        "train_log_path": TRAIN_LOG_PATH,
        "install_log_path": INSTALL_LOG_PATH,
        "outputs_dir": OUTPUTS_DIR,
        "run_tag": RUN_TAG,
    },
) as post_run:
    outputs_artifact = wandb.Artifact(
        name=f"musicgen-outputs-{post_run.id}",
        type="model-output",
        description="AudioCraft output directory, installation log and training stdout log.",
        metadata={
            "latest_xp_dir": str(latest_xp_dir) if latest_xp_dir else None,
            "has_train_log": os.path.exists(TRAIN_LOG_PATH),
            "has_install_log": os.path.exists(INSTALL_LOG_PATH),
            "run_tag": RUN_TAG,
        },
    )

    if os.path.exists(TRAIN_LOG_PATH):
        outputs_artifact.add_file(TRAIN_LOG_PATH)

    if os.path.exists(INSTALL_LOG_PATH):
        outputs_artifact.add_file(INSTALL_LOG_PATH)

    if latest_xp_dir is not None and latest_xp_dir.exists():
        outputs_artifact.add_dir(str(latest_xp_dir), name="xp")

    post_run.log_artifact(outputs_artifact, aliases=["latest", "train-output"])

print("\n✓ Всё завершено")
