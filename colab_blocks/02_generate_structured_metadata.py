import os
import json
import re
import gc
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from tqdm import tqdm


gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()


OUTPUT_DIR = "/content/musiccaps_data"
dataset = load_dataset("google/MusicCaps", split="train")

print("Загружаем локальную Llama 3...")
model_id = "NousResearch/Meta-Llama-3-8B-Instruct"

quantization_config = BitsAndBytesConfig(load_in_4bit=True)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map={"": 0},
    quantization_config=quantization_config,
    torch_dtype=torch.float16,
)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    pad_token_id=tokenizer.eos_token_id
)

def process_metadata_llama(row):
    ytid = row['ytid']
    caption = row['caption']
    json_path = os.path.join(OUTPUT_DIR, f"{ytid}.json")
    wav_path = os.path.join(OUTPUT_DIR, f"{ytid}.wav")

    
    if os.path.exists(json_path) or not os.path.exists(wav_path):
        return

    messages = [
        {"role": "system", "content": "You are a data extractor. Output ONLY valid JSON."},
        {"role": "user", "content": f"Extract this music description into JSON.\nSchema: {{\"description\": \"string\", \"general_mood\": \"string\", \"genre_tags\": [\"string\"], \"lead_instrument\": \"string\", \"accompaniment\": \"string\", \"tempo_and_rhythm\": \"string\", \"vocal_presence\": \"string\", \"production_quality\": \"string\"}}\n\nDescription: {caption}"}
    ]

    prompt = generator.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    try:
        outputs = generator(prompt, max_new_tokens=1024, temperature=0.1, do_sample=False, return_full_text=False)
        response_text = outputs[0]['generated_text'].strip()

       
        response_text = response_text.replace("```json", "").replace("```", "").strip()

        
        if response_text.startswith("{") and not response_text.endswith("}"):
            response_text += "\n}"

        
        match = re.search(r'\{.*\}', response_text, re.DOTALL)

        if match:
            json_string = match.group(0)
            structured_data = json.loads(json_string)

            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(structured_data, f, indent=2, ensure_ascii=False)
        else:
            print(f"\n[-] Модель выдала ошибку для {ytid}:\n{response_text}")

    except json.JSONDecodeError as e:
        print(f"\n[-] Кривой JSON у {ytid}. Текст от модели:\n{json_string}")
    except Exception as e:
        print(f"\n[-] Непредвиденная ошибка на {ytid}: {e}")

print("Начинаем генерацию JSON. .")
for row in tqdm(dataset, desc="Генерация JSON "):
    process_metadata_llama(row)
