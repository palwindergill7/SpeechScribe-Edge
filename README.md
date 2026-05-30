# SpeechScribe Edge

![SpeechScribe Edge](assets/SpeechScribe%20Edge.png)

Transcribe audio with Whisper, translate, edit, and export to PDF.

## Features
- **File tab** — upload or record audio, pick a Whisper model, adjust chunk/stride, optionally translate into 14 languages, export
- **Real-time tab** — live mic streaming with adjustable chunk and stride; transcript updates every few seconds

## Project layout
```
SpeechScribe-Edge/
├── app.py                        # Gradio UI (two tabs)
├── transcriber.py                # Whisper (file mode)
├── translator.py                 # NLLB-200 translation
├── exporters.py                  # PDF / DOCX / TXT / SRT
├── requirements.txt
├── real_time/
│   └── real_time_transcriber.py  # Whisper (real-time mic mode)
└── outputs/                      # generated files
```

## Setup
```bash
conda create -n speechscribe python=3.11 -y && conda activate speechscribe
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
# also install ffmpeg: winget install ffmpeg (Windows) / brew install ffmpeg (macOS)
```

## Run
```bash
python app.py
```
Open `http://127.0.0.1:7860` in your browser.

## Notes
- Models are downloaded from Hugging Face on first run and cached locally.
- Use `whisper-tiny` or `whisper-base` in the real-time tab for lowest latency.
- To add a language add a `"Name": "flores_code"` entry in `translator.py`.
