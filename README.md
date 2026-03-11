# Домашнее задание 4: Fine-tuning MusicGen

В данном репозитории представлена реализация пайплайна для дообучения (fine-tuning) модели MusicGen из библиотеки AudioCraft на датасете MusicCaps с использованием обогащенных структурированных метаданных. 

---

## 🛠 1. Как собрать и подготовить датасет

Пайплайн подготовки данных разбит на два этапа, которые выполняются последовательно:

**Шаг 1. Загрузка аудиофрагментов (MusicCaps)**
Для загрузки аудио используется автоматизированный скрипт на базе библиотеки `datasets`. 
* Во избежание скачивания видео целиком, скрипт использует `yt-dlp -g -f bestaudio` для получения прямой ссылки на аудиопоток.
* Затем вызывается `ffmpeg -ss [start_s] -to [end_s]` для точечного скачивания нужного 10-секундного фрагмента.
* Файлы сохраняются в формате `.wav` (32000 Hz, 1 канал) в директорию `musiccaps_data`.

**Шаг 2. Парсинг и структурирование метаданных (LLM)**
Оригинальные текстовые описания (captions) переводятся в строгий JSON-формат для улучшения качества обучения.
* Используется локальная модель `NousResearch/Meta-Llama-3-8B-Instruct` в 4-bit квантовании.
* LLM извлекает из текста значения для заданных полей: `description`, `general_mood`, `genre_tags`, `lead_instrument`, `accompaniment`, `tempo_and_rhythm`, `vocal_presence`, `production_quality`.
* Ответы модели парсятся с помощью регулярных выражений и сохраняются в `.json` файлы с тем же именем, что и исходный `.wav`.

---

## 🚀 2. Как запустить процесс обучения

**Шаг 1. Подготовка окружения и репозитория**
* Склонируйте данный репозиторий. В нем уже модифицирован файл `audiocraft/data/music_dataset.py` (в датакласс `MusicInfo` добавлены кастомные поля из JSON-схемы, обновлена логика парсинга атрибутов).
* Создайте изолированное окружение (рекомендуется `micromamba` с Python 3.10) и установите зависимости, включая PyTorch под вашу версию CUDA и пропатченный AudioCraft (`pip install -e .`).

**Шаг 2. Сборка манифестов**
Скрипт подготовки манифестов считывает пары `.wav` и `.json`, нормализует данные и формирует файлы `train.jsonl.gz` и `valid.jsonl.gz` в директории `egs/musiccaps/`. Для датасета создается конфигурационный файл Hydra (`musiccaps_ft.yaml`).

**Шаг 3. Запуск файнтюнинга**
Обучение запускается через менеджер `dora` командой из корня репозитория:

```bash
dora run solver=musicgen/musicgen_base_32khz \
  dset=audio/musiccaps_ft \
  model/lm/model_scale=small \
  continue_from=//pretrained/facebook/musicgen-small \
  ++conditioner.text.merge_text_p=0.25 \
  ++conditioner.text.drop_desc_p=0.5 \
  ++conditioner.text.drop_other_p=0.5 \
  ++conditioner.text.text_attributes=[genre_tags,general_mood,lead_instrument,accompaniment,tempo_and_rhythm,vocal_presence,production_quality] \
  optim.epochs=50 \
  optim.updates_per_epoch=150 \
  optim.lr=8e-6
Важно: Параметры drop_desc_p=0.5 и drop_other_p=0.5 обеспечивают зануление части текстовых условий в процессе обучения, что сохраняет способность модели к Classifier-Free Guidance (CFG).

🎧 3. Как запустить инференс
Шаг 1. Экспорт обученных весов
Запустите скрипт экспорта: он извлечет best_state из файла checkpoint.th (сохраненного dora) и конвертирует его в state_dict.bin и compression_state_dict.bin в папку exported_musicgen_ft.

Шаг 2. Генерация аудио
Скрипт инференса выполняет следующие действия:

Загружает 5 тестовых структурированных JSON-промптов.

Форматирует каждый JSON в текстовую строку вида ключ: значение, в точности повторяя структуру, которую модель видела при обучении.

Загружает веса через MusicGen.get_pretrained('exported_musicgen_ft').

Генерирует аудио длительностью 12 секунд (duration=12, cfg_coef=3.0).

Сохраняет результаты в папку generated_prompts под именами prompt_1.wav ... prompt_5.wav.

📄 Краткий отчет по работе
1. С какими трудностями столкнулись?
Настройка окружения: Потребовалась тонкая настройка зависимостей (специфические версии torchaudio, xformers) и использование micromamba для корректной работы AudioCraft на современных GPU.

Модификация AudioCraft: Необходимо было аккуратно внедрить новые поля в датакласс MusicInfo и обновить методы attribute_getter и is_valid_field, чтобы списки (например, genre_tags) корректно обрабатывались через get_keyword_list.

Парсинг LLM: Локальные модели иногда генерируют лишний текст помимо JSON. Проблема была решена написанием надежного парсера на базе регулярных выражений (re.search(r'\{.*\}', ...)).

2. LLM и промпты
Для обогащения метаданных использовалась локальная модель NousResearch/Meta-Llama-3-8B-Instruct (4-bit квантование).

Системный промпт: You are a data extractor. Output ONLY valid JSON.

Пользовательский промпт: Extract this music description into JSON. Schema: {"description": "string", "general_mood": "string", "genre_tags": ["string"], "lead_instrument": "string", "accompaniment": "string", "tempo_and_rhythm": "string", "vocal_presence": "string", "production_quality": "string"}\n\nDescription: {caption}

3. Гиперпараметры обучения
Модель: musicgen-small

Learning rate: 8e-6

Epochs: 50

Updates per epoch: 150

Batch size: 1

Segment duration: 2 секунды
