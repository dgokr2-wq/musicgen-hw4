# Run Notes

## Project
Fine-tuning MusicGen on MusicCaps with structured metadata fields.

## Dataset preparation
The dataset was built from MusicCaps:
- audio was downloaded from YouTube using `yt-dlp` + `ffmpeg`
- each sample was saved as a 10-second `.wav` file
- captions were converted into structured JSON metadata using a local Llama 3 model

## Structured metadata schema
Each audio file was paired with JSON metadata in the following format:
- `description`
- `general_mood`
- `genre_tags`
- `lead_instrument`
- `accompaniment`
- `tempo_and_rhythm`
- `vocal_presence`
- `production_quality`

## AudioCraft modifications
The file `audiocraft/data/music_dataset.py` was modified to:
- add new fields into `MusicInfo`
- support parsing of `genre_tags` as keyword list
- support parsing of the new string metadata fields
- include the new fields in text augmentation / conditioning

## Environment
- Python: 3.10
- Environment manager: micromamba
- AudioCraft installed in editable mode
- GPU compatibility fix applied for Blackwell architecture using a newer PyTorch stack

## GPU / framework notes
The original PyTorch stack was incompatible with the Blackwell GPU (`sm_120`), which caused:
- `CUDA error: no kernel image is available for execution on the device`

This was fixed by switching to:
- `torch==2.7.1`
- `torchvision==0.22.1`
- `torchaudio==2.7.1`
- CUDA 12.8 wheels
- `xformers==0.0.31`

## Training setup
Base model:
- `facebook/musicgen-small`

Compression model:
- `facebook/encodec_32khz`

Main training parameters:
- epochs: `50`
- updates per epoch: `150`
- learning rate: `8e-6`
- batch size: `1`
- num workers: `0`
- segment duration: `2`
- autocast: `true`
- autocast dtype: `float16`
- EMA: disabled

Conditioning / text settings:
- `merge_text_p=0.25`
- `drop_desc_p=0.5`
- `drop_other_p=0.5`

Text attributes used for conditioning:
- `genre_tags`
- `general_mood`
- `lead_instrument`
- `accompaniment`
- `tempo_and_rhythm`
- `vocal_presence`
- `production_quality`

## Logging
During training:
- internal AudioCraft W&B logging was disabled to avoid deadlocks / runtime issues
- TensorBoard logging remained enabled

After training:
- outputs, logs, and generated results were uploaded separately to W&B

## Inference notes
The fine-tuned checkpoint was exported manually because the default AudioCraft export path failed with newer PyTorch due to `torch.load(weights_only=True)` behavior.

For inference:
- the exported LM was loaded through `MusicGen.get_pretrained(...)`
- 5 evaluation prompts were saved into `test_prompts.json`
- generation duration was set to `12` seconds
- prompts were generated one by one to reduce VRAM pressure

Generation parameters:
- duration: `12`
- `top_k=250`
- `temperature=1.0`
- `cfg_coef=3.0`

## Final artifacts
The following outputs were produced:
- exported fine-tuned weights
- `prompt_1.wav`
- `prompt_2.wav`
- `prompt_3.wav`
- `prompt_4.wav`
- `prompt_5.wav`
- logs for training and generation
