# 🎵 MusicGen Fine-Tuning (Домашнее задание 4)

В данном репозитории представлена реализация пайплайна для дообучения (fine-tuning) модели MusicGen на датасете MusicCaps с использованием структурированных метаданных. 

Репозиторий содержит модифицированный код `AudioCraft`, скрипты для парсинга данных, запуска обучения и инференса.

---

## 🛠 1. Как собрать и подготовить датасет

Пайплайн подготовки данных состоит из двух этапов: скачивание аудио и обогащение метаданных.

**Шаг 1. Скачивание аудио (MusicCaps)**
Для загрузки аудио используется скрипт, работающий через `yt-dlp` и `ffmpeg`. 
Видео не скачиваются целиком: `yt-dlp` извлекает прямую ссылку на аудиопоток, а `ffmpeg` вырезает нужный 10-секундный фрагмент и конвертирует его в формат `.wav` (32000 Hz, 1 channel).

**Шаг 2. Обогащение метаданных (LLM)**
Сырые текстовые описания переводятся в строгий JSON-формат. 
Для этого применяется локальная модель `Meta-Llama-3-8B-Instruct` (в 4-bit квантовании). Скрипт генерирует `.json` файлы для каждого трека и сохраняет их рядом с `.wav` файлами в папке датасета. Обязуемая схема включает поля: `general_mood`, `genre_tags`, `lead_instrument`, `accompaniment`, `tempo_and_rhythm`, `vocal_presence`, `production_quality`.

---

## 🚀 2. Как запустить процесс обучения

Перед запуском обучения необходимо убедиться, что окружение собрано (рекомендуется использовать `micromamba` с Python 3.10 и PyTorch, совместимым с вашей архитектурой GPU, например CUDA 12.8 для Blackwell).

В коде `AudioCraft` (в файле `audiocraft/data/music_dataset.py`) уже добавлены новые поля в датакласс `MusicInfo` и обновлена логика парсинга атрибутов.

Для старта файнтюнинга со сборкой манифестов и запуском `dora run`, выполните основной тренировочный скрипт:

```bash
# Пример команды, которая запускается внутри скрипта:
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
Настройки drop_desc_p и drop_other_p гарантируют, что модель с вероятностью 50% будет обучаться без текста, что сохраняет способность к Classifier-Free Guidance (CFG).

🎧 3. Как запустить инференс
После завершения обучения веса сохраняются в папку экспериментов (XP dir).

Скрипт инференса автоматически:

Экспортирует best_state в формат, понятный классу MusicGen (state_dict.bin).

Загружает кастомные промпты (5 тестовых заданий из ТЗ).

Форматирует JSON-промпты в строку вида ключ: значение, идентичную той, что модель видела при обучении.

Генерирует аудио длительностью 12 секунд.

Все 5 сгенерированных треков (prompt_1.wav ... prompt_5.wav) сохраняются в выходную директорию.

📄 Краткий отчет по работе
1. С какими трудностями столкнулись?
Основная сложность заключалась в настройке окружения для корректной работы AudioCraft на современных GPU (потребовалась установка специфических версий torchaudio, xformers и использование micromamba для изоляции зависимостей). Также потребовалось аккуратно модифицировать music_dataset.py, чтобы модель корректно подхватывала новые кастомные поля и обрабатывала списки (например, genre_tags) через get_keyword_list.

2. Какую LLM использовали для парсинга и какой системный промпт сработал лучше всего?
Использовалась локальная модель NousResearch/Meta-Llama-3-8B-Instruct в 4-bit квантовании.
Системный промпт: You are a data extractor. Output ONLY valid JSON.
Пользовательский промпт: Extract this music description into JSON. Schema: {"description": "string", "general_mood": "string", ...}
Плюс был написан парсер на основе регулярных выражений (re.search(r'\{.*\}', response_text, re.DOTALL)), чтобы надежно извлекать JSON из ответов модели.

3. Какие гиперпараметры обучения вы использовали?

Learning Rate: 8e-6

Epochs: 50

Updates per epoch: 150

Batch size: 1

Segment duration: 2 секунды
