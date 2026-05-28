# 🎙️ Audio → PDF

Upload or record audio, transcribe it with OpenAI Whisper, optionally translate
it with NLLB-200, edit the text, then export to **PDF / DOCX / TXT / SRT**.
Built with Gradio, GPU-aware (uses CUDA + fp16 automatically when available).

## Features
- Upload an audio file **or** record from the microphone
- Whisper model selector (tiny → large-v3) to trade speed vs. accuracy
- Optional timestamps in the transcript
- Editable transcript before export
- Translation into 14 languages (extend in `translator.py`)
- Multiple output formats in one click (PDF, DOCX, TXT, SRT)

## Project layout
```
audio-to-pdf/
├── app.py          # Gradio UI + wiring
├── transcriber.py  # Whisper speech-to-text
├── translator.py   # NLLB translation + language list
├── exporters.py    # PDF / DOCX / TXT / SRT writers
├── requirements.txt
└── outputs/        # generated files (created on first run)
```

## Setup (your GPU machine)

1. (Recommended) create an environment:
   ```bash
   conda create -n audio2pdf python=3.11 -y
   conda activate audio2pdf
   ```

2. Install PyTorch with CUDA matching your machine — see
   https://pytorch.org/get-started/locally/ . For example:
   ```bash
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
   ```

3. Install the rest:
   ```bash
   pip install -r requirements.txt
   ```

4. Install **ffmpeg** (Whisper's pipeline uses it to decode audio):
   - Windows: `winget install ffmpeg` or `choco install ffmpeg`
   - Linux: `sudo apt install ffmpeg`
   - macOS: `brew install ffmpeg`

## Run
```bash
python app.py
```
Then open the local URL Gradio prints (usually http://127.0.0.1:7860).

To expose a temporary public link, change the last line of `app.py` to
`demo.launch(share=True)`.

## Notes
- First run downloads the Whisper and NLLB models from Hugging Face; later runs
  are cached.
- Switching the Whisper model in the UI reloads it on the next transcription.
- `large-v3` needs a fair amount of VRAM; if you hit OOM, pick a smaller model.
- To add a language, add a `"Name": "flores_code"` entry to `LANGUAGES` in
  `translator.py`.
