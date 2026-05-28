"""
Audio-to-PDF: transcribe audio (Whisper), optionally translate (NLLB),
and export to PDF / DOCX / TXT / SRT. Gradio UI, GPU-aware.
"""

import gradio as gr

from transcriber import Transcriber
from translator import Translator, LANGUAGES
from exporters import export_file

# Lazy singletons so the UI loads instantly and models load on first use.
_transcriber = None
_translator = None


def get_transcriber(model_name: str) -> Transcriber:
    global _transcriber
    if _transcriber is None or _transcriber.model_name != model_name:
        _transcriber = Transcriber(model_name)
    return _transcriber


def get_translator() -> Translator:
    global _translator
    if _translator is None:
        _translator = Translator()
    return _translator


def transcribe_audio(audio_path, model_name, with_timestamps, progress=gr.Progress()):
    """Step 1: audio -> raw transcript text."""
    if not audio_path:
        raise gr.Error("Please upload an audio file or record from the mic first.")
    progress(0.1, desc="Loading model...")
    tr = get_transcriber(model_name)
    progress(0.4, desc="Transcribing...")
    text = tr.transcribe(audio_path, return_timestamps=with_timestamps)
    progress(1.0, desc="Done")
    return text


def translate_text(text, target_lang):
    """Step 2 (optional): translate the (possibly edited) transcript."""
    if not text or not text.strip():
        raise gr.Error("There's no transcript to translate yet.")
    if target_lang == "None (keep original)":
        return text
    tl = get_translator()
    return tl.translate(text, LANGUAGES[target_lang])


def make_files(text, formats, base_name):
    """Step 3: render the final text into the chosen file formats."""
    if not text or not text.strip():
        raise gr.Error("There's no text to export.")
    if not formats:
        raise gr.Error("Pick at least one output format.")
    base = (base_name or "transcript").strip() or "transcript"
    return [export_file(text, fmt, base) for fmt in formats]


with gr.Blocks(title="Audio to PDF", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 🎙️ Audio → Text → Document\n"
        "Upload or record audio, transcribe it with Whisper, optionally "
        "translate, edit the text, then export to PDF / DOCX / TXT / SRT."
    )

    with gr.Row():
        with gr.Column(scale=1):
            audio_in = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Audio input",
            )
            model_dd = gr.Dropdown(
                choices=[
                    "openai/whisper-tiny",
                    "openai/whisper-base",
                    "openai/whisper-small",
                    "openai/whisper-medium",
                    "openai/whisper-large-v3",
                ],
                value="openai/whisper-small",
                label="Whisper model (bigger = slower, more accurate)",
            )
            ts_chk = gr.Checkbox(label="Include timestamps", value=False)
            transcribe_btn = gr.Button("1 · Transcribe", variant="primary")

        with gr.Column(scale=2):
            transcript = gr.Textbox(
                label="Transcript (editable)",
                lines=14,
                placeholder="Transcribed text will appear here. You can edit it before exporting.",
            )
            with gr.Row():
                target_dd = gr.Dropdown(
                    choices=["None (keep original)"] + list(LANGUAGES.keys()),
                    value="None (keep original)",
                    label="Translate to",
                )
                translate_btn = gr.Button("2 · Translate")

            with gr.Row():
                formats = gr.CheckboxGroup(
                    choices=["pdf", "docx", "txt", "srt"],
                    value=["pdf"],
                    label="Output formats",
                )
                name_box = gr.Textbox(label="File name (no extension)", value="transcript")
            export_btn = gr.Button("3 · Generate files", variant="primary")
            files_out = gr.File(label="Download", file_count="multiple")

    transcribe_btn.click(
        transcribe_audio,
        inputs=[audio_in, model_dd, ts_chk],
        outputs=transcript,
    )
    translate_btn.click(
        translate_text,
        inputs=[transcript, target_dd],
        outputs=transcript,
    )
    export_btn.click(
        make_files,
        inputs=[transcript, formats, name_box],
        outputs=files_out,
    )


if __name__ == "__main__":
    demo.launch()
