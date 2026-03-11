

import os
import json
from pathlib import Path


GEN_ENV = RUNTIME_ENV.copy()
GEN_ENV["USER"] = "root"
GEN_ENV["PYTHONPATH"] = AUDIOCRAFT_DIR + (":" + GEN_ENV["PYTHONPATH"] if "PYTHONPATH" in GEN_ENV else "")
GEN_ENV["AUDIOCRAFT_DORA_DIR"] = OUTPUTS_DIR


assert latest_xp_dir is not None and Path(latest_xp_dir).exists(), "latest_xp_dir не найден. Сначала заверши train."
latest_xp_dir = Path(latest_xp_dir)
checkpoint_path = latest_xp_dir / "checkpoint.th"
assert checkpoint_path.exists(), f"Не найден checkpoint: {checkpoint_path}"

EXPORT_DIR = Path(OUTPUTS_DIR) / "exported_musicgen_ft"
GEN_DIR = Path(OUTPUTS_DIR) / "generated_prompts"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
GEN_DIR.mkdir(parents=True, exist_ok=True)

print("latest_xp_dir =", latest_xp_dir)
print("checkpoint_path =", checkpoint_path)
print("EXPORT_DIR =", EXPORT_DIR)
print("GEN_DIR =", GEN_DIR)


export_code = f"""
from pathlib import Path
import torch
from omegaconf import OmegaConf
from audiocraft import __version__

checkpoint_path = Path(r"{str(checkpoint_path)}")
export_dir = Path(r"{str(EXPORT_DIR)}")
export_dir.mkdir(parents=True, exist_ok=True)

lm_out = export_dir / "state_dict.bin"
compression_out = export_dir / "compression_state_dict.bin"

# ВАЖНО: для нового torch нужен weights_only=False,
# так как checkpoint доверенный и содержит DictConfig
pkg = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

if pkg.get("fsdp_best_state"):
    best_state = pkg["fsdp_best_state"]["model"]
else:
    assert pkg.get("best_state"), "В checkpoint нет best_state"
    best_state = pkg["best_state"]["model"]

new_pkg = {{
    "best_state": best_state,
    "xp.cfg": OmegaConf.to_yaml(pkg["xp.cfg"]),
    "version": __version__,
    "exported": True,
}}

torch.save(new_pkg, lm_out)

# Pointer на pretrained compression model
compression_pkg = {{
    "pretrained": "facebook/encodec_32khz",
    "exported": True,
    "version": __version__,
}}
torch.save(compression_pkg, compression_out)

print("LM exported to:", lm_out)
print("Compression pointer exported to:", compression_out)
print("Export dir ready:", export_dir)
"""

run_live(
    [MAMBA_BIN, "run", "-n", ENV_NAME, "python", "-c", export_code],
    env=GEN_ENV
)


test_prompts = [
    {
      "description": "An epic and triumphant orchestral soundtrack featuring powerful brass and a sweeping string ensemble, driven by a fast march-like rhythm and an epic background choir, recorded with massive stadium reverb.",
      "general_mood": "Epic, heroic, triumphant, building tension",
      "genre_tags": ["Cinematic", "Orchestral", "Soundtrack"],
      "lead_instrument": "Powerful brass section (horns, trombones)",
      "accompaniment": "Sweeping string ensemble, heavy cinematic percussion, timpani",
      "tempo_and_rhythm": "Fast, driving, march-like rhythm",
      "vocal_presence": "Epic choir in the background (wordless chanting)",
      "production_quality": "High fidelity, wide stereo image, massive stadium reverb"
    },
    {
      "description": "A relaxing lo-fi hip-hop instrumental with a muffled electric piano playing jazz chords over a dusty vinyl crackle, deep sub-bass, and a slow boom-bap drum loop.",
      "general_mood": "Relaxing, nostalgic, chill, melancholic",
      "genre_tags": ["Lo-Fi Hip Hop", "Chillhop", "Instrumental"],
      "lead_instrument": "Muffled electric piano (Rhodes) playing jazz chords",
      "accompaniment": "Dusty vinyl crackle, deep sub-bass, soft boom-bap drum loop",
      "tempo_and_rhythm": "Slow, laid-back, swinging groove",
      "vocal_presence": "None",
      "production_quality": "Lo-Fi, vintage, warm tape saturation, slightly muffled high frequencies"
    },
    {
      "description": "An energetic progressive house dance track with a bright detuned synthesizer lead, pumping sidechain bass, and chopped vocal samples over a fast four-on-the-floor beat.",
      "general_mood": "Energetic, uplifting, party vibe, euphoric",
      "genre_tags": ["EDM", "Progressive House", "Dance"],
      "lead_instrument": "Bright, detuned synthesizer lead",
      "accompaniment": "Pumping sidechain bass, risers, crash cymbals",
      "tempo_and_rhythm": "Fast, driving, strict four-on-the-floor beat",
      "vocal_presence": "Chopped vocal samples used as a rhythmic instrument",
      "production_quality": "Modern, extremely loud, punchy, club-ready mix"
    },
    {
      "description": "An intimate acoustic folk instrumental featuring a fingerpicked acoustic guitar, light tambourine, and subtle upright bass, played in a gentle waltz-like rhythm.",
      "general_mood": "Intimate, warm, acoustic, peaceful",
      "genre_tags": ["Folk", "Acoustic", "Indie"],
      "lead_instrument": "Fingerpicked acoustic guitar",
      "accompaniment": "Light tambourine, subtle upright bass, distant ambient room sound",
      "tempo_and_rhythm": "Mid-tempo, gentle, waltz-like triple meter",
      "vocal_presence": "None",
      "production_quality": "Raw, organic, close-mic recording, natural room acoustics"
    },
    {
      "description": "A dark cyberpunk synthwave instrumental driven by an aggressive distorted analog bass synthesizer, arpeggiated synth plucks, and a retro 80s drum machine.",
      "general_mood": "Dark, futuristic, gritty, mysterious",
      "genre_tags": ["Synthwave", "Cyberpunk", "Darkwave"],
      "lead_instrument": "Aggressive, distorted analog bass synthesizer",
      "accompaniment": "Arpeggiated synth plucks, retro 80s drum machine (gated snare)",
      "tempo_and_rhythm": "Driving, mid-tempo, robotic precision",
      "vocal_presence": "None",
      "production_quality": "Retro-futuristic, heavy compression, synthetic, 80s aesthetic"
    }
]

prompts_json_path = GEN_DIR / "test_prompts.json"
with open(prompts_json_path, "w", encoding="utf-8") as f:
    json.dump(test_prompts, f, ensure_ascii=False, indent=2)

print("Saved prompts JSON:", prompts_json_path)


generate_code = f"""
import json
from pathlib import Path
import torch
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

export_dir = Path(r"{str(EXPORT_DIR)}")
gen_dir = Path(r"{str(GEN_DIR)}")
prompts_path = Path(r"{str(prompts_json_path)}")

with open(prompts_path, "r", encoding="utf-8") as f:
    prompts = json.load(f)

def structured_prompt_to_text(p):
    fields = [
        ("description", p["description"]),
        ("general_mood", p["general_mood"]),
        ("genre_tags", ", ".join(p["genre_tags"])),
        ("lead_instrument", p["lead_instrument"]),
        ("accompaniment", p["accompaniment"]),
        ("tempo_and_rhythm", p["tempo_and_rhythm"]),
        ("vocal_presence", p["vocal_presence"]),
        ("production_quality", p["production_quality"]),
    ]
    return "\\n".join([f"{{k}}: {{v}}" for k, v in fields])

# PATCH: у fine-tuned модели max_duration маленький,
# а MusicGen по умолчанию пытается поставить extend_stride=18
_orig_set_generation_params = MusicGen.set_generation_params

def _safe_set_generation_params(self, *args, **kwargs):
    max_d = float(self.max_duration)
    safe_stride = min(1.5, max(0.5, max_d - 0.01))

    if "extend_stride" not in kwargs or kwargs["extend_stride"] is None:
        kwargs["extend_stride"] = safe_stride
    else:
        kwargs["extend_stride"] = min(float(kwargs["extend_stride"]), safe_stride)

    return _orig_set_generation_params(self, *args, **kwargs)

MusicGen.set_generation_params = _safe_set_generation_params

print("Loading exported fine-tuned model from:", export_dir)
model = MusicGen.get_pretrained(str(export_dir))
print("Loaded. model.max_duration =", model.max_duration)

safe_stride = min(1.5, max(0.5, float(model.max_duration) - 0.01))

model.set_generation_params(
    duration=12,
    use_sampling=True,
    top_k=250,
    top_p=0.0,
    temperature=1.0,
    cfg_coef=3.0,
    extend_stride=safe_stride,
)

print("Using extend_stride =", safe_stride)

# Генерируем по одному треку, чтобы не перегружать VRAM
for idx, p in enumerate(prompts, 1):
    txt = structured_prompt_to_text(p)

    print("\\n" + "=" * 80)
    print(f"PROMPT {{idx}}")
    print(txt)

    wav = model.generate([txt])[0]

    out_stem = gen_dir / f"prompt_{{idx}}"
    audio_write(
        str(out_stem),
        wav.cpu(),
        model.sample_rate,
        strategy="loudness",
        loudness_compressor=True
    )
    print("Saved:", str(out_stem) + ".wav")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("\\nAll files saved to:", gen_dir)
"""

run_live(
    [MAMBA_BIN, "run", "-n", ENV_NAME, "python", "-c", generate_code],
    env=GEN_ENV,
    log_path=str(GEN_DIR / "generation_stdout.log")
)


try:
    with wandb.init(
        project=WANDB_PROJECT,
        group=WANDB_GROUP,
        name=f"musicgen-inference-{RUN_TAG}",
        job_type="inference",
        config={
            "export_dir": str(EXPORT_DIR),
            "generated_dir": str(GEN_DIR),
            "checkpoint_path": str(checkpoint_path),
            "duration_sec": 12,
            "num_prompts": 5,
        },
    ) as infer_run:
        infer_artifact = wandb.Artifact(
            name=f"musicgen-generated-prompts-{infer_run.id}",
            type="inference-output",
            description="Five generated wav files for the Homework 4 evaluation prompts."
        )
        infer_artifact.add_dir(str(GEN_DIR), name="generated")
        infer_run.log_artifact(infer_artifact, aliases=["latest", "prompt-wavs"])
    print("✓ W&B inference artifact uploaded")
except Exception as e:
    print("W&B upload skipped:", e)

print("\\n✓ PART 2 COMPLETED")
print("Generated WAV folder:", GEN_DIR)
print("Exported model folder:", EXPORT_DIR)
