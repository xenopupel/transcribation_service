# Transcribe Service

Сервис транскрибации стерео звонков на базе локального форка GigaAM.

Что делает:

- принимает только стерео аудио;
- режет звонок VAD по каналам;
- батчит сегменты внутри одного аудио с сортировкой по длине;
- распознает речь через GigaAM;
- применяет постпроцессинг IVR/HOLD;
- маскирует персональные данные через spaCy;
- отдает результат в `.txt` и `.json`;
- в UI можно загрузить несколько файлов и скачать один ZIP с `.txt`.

## Установка

Поставить зависимости проекта:

```powershell
pip install -r requirements.txt
python -m spacy download ru_core_news_lg
```

Подтянуть форк GigaAM:

```powershell
git clone https://github.com/xenopupel/GigaAM.git
```

Поставить GigaAM:

```powershell
cd GigaAM
pip install -e ".[torch]"
```

## UI

UI поддерживает:

- drag-and-drop аудиофайлов;
- форматы `.flac`, `.m4a`, `.mp3`, `.ogg`, `.wav`;
- общий статус загрузки, очереди и обработки;
- отображение позиции в очереди;
- скачивание ZIP с `.txt` результатами после полной обработки всех файлов.

Файлы внутри ZIP получают имена исходных аудио, но с расширением `.txt`.

## API

Создать job и сразу поставить в очередь:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/jobs" `
  -F "file=@Q:\path\to\audio.mp3"
```

Загрузить файл без старта обработки:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/jobs?enqueue=false" `
  -F "file=@Q:\path\to\audio.mp3"
```

Запустить загруженные job:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/jobs/start" `
  -H "Content-Type: application/json" `
  -d "{\"job_ids\":[\"<job_id_1>\",\"<job_id_2>\"]}"
```

Проверить статус:

```powershell
curl.exe "http://127.0.0.1:8000/jobs/<job_id>"
```

Скачать TXT:

```powershell
curl.exe "http://127.0.0.1:8000/jobs/<job_id>/result.txt" -o result.txt
```

Скачать ZIP по нескольким job:

```powershell
curl.exe -L "http://127.0.0.1:8000/jobs/archive?job_id=<job_id_1>&job_id=<job_id_2>" `
  -o transcripts.zip
```

По умолчанию IVR не попадает в результат. Чтобы включить IVR:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/jobs?include_ivr=true" `
  -F "file=@Q:\path\to\audio.mp3"
```

## Настройки

Пример настроек лежит в `.env.example`:

```env
TRANSCRIBE_STORAGE_DIR=storage
TRANSCRIBE_DEVICE=cuda
TRANSCRIBE_MAX_UPLOAD_MB=500
TRANSCRIBE_MIN_FREE_DISK_MB=2048
TRANSCRIBE_MAX_QUEUED_JOBS=100
TRANSCRIBE_BATCH_SIZE=8
TRANSCRIBE_MASK_PII=true
TRANSCRIBE_SPACY_MODEL=ru_core_news_lg
```

Количество GPU-воркеров сейчас не настраивается. В сервисе используется один GPU worker, поэтому аудио обрабатываются по очереди. Это сделано намеренно, чтобы не упираться в память видеокарты.

## Storage

`storage/uploads/` хранит исходные аудио только пока job ожидает обработки или обрабатывается.

После `done` или `failed` исходный upload удаляется. Результаты остаются в:

```text
storage/results/<job_id>/
```

SQLite база job:

```text
storage/jobs.sqlite3
```

## CLI

Один файл:

```powershell
python -m gigaam_service.process_file `
  --audio "Q:\path\to\audio.mp3" `
  --device cuda `
  --output-json out\result.json `
  --output-txt out\result.txt
```

Папка:

```powershell
python scripts\batch_transcribe_folder.py `
  --input-dir "Q:\path\to\audio_folder" `
  --output-dir out `
  --device cuda
```