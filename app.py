"""
Audio-to-PDF: transcribe audio (Whisper), optionally translate (NLLB),
and export to PDF / DOCX / TXT / SRT. Gradio UI, GPU-aware.

Real-time tab: streams microphone input, buffers 3-second chunks,
and appends Whisper transcriptions live.
"""

import numpy as np
import gradio as gr

from transcriber import Transcriber
from translator import Translator, LANGUAGES
from exporters import export_file
from real_time.real_time_transcriber import RealtimeTranscriber, CHUNK_SECONDS

# ---------------------------------------------------------------------------
# Lazy model singletons — loaded on first use, not at startup
# ---------------------------------------------------------------------------

_transcriber: Transcriber | None = None
_translator: Translator | None = None
_rt_transcriber: RealtimeTranscriber | None = None


def _get_transcriber(model_name: str) -> Transcriber:
    global _transcriber
    if _transcriber is None or _transcriber.model_name != model_name:
        _transcriber = Transcriber(model_name)
    return _transcriber


def _get_translator() -> Translator:
    global _translator
    if _translator is None:
        _translator = Translator()
    return _translator


def _get_rt_transcriber(model_name: str) -> RealtimeTranscriber:
    global _rt_transcriber
    if _rt_transcriber is None or _rt_transcriber.model_name != model_name:
        _rt_transcriber = RealtimeTranscriber(model_name)
    return _rt_transcriber


# ---------------------------------------------------------------------------
# File-transcription handlers (Step 1 / 2 / 3)
# ---------------------------------------------------------------------------

def transcribe_audio(audio_path, model_name, with_timestamps, chunk_length_s, stride_length_s, progress=gr.Progress()):
    if not audio_path:
        raise gr.Error("Please upload an audio file or record from the mic first.")
    progress(0.1, desc="Loading model...")
    tr = _get_transcriber(model_name)
    progress(0.4, desc="Transcribing...")
    text = tr.transcribe(
        audio_path,
        return_timestamps=with_timestamps,
        chunk_length_s=int(chunk_length_s),
        stride_length_s=int(stride_length_s),
    )
    progress(1.0, desc="Done")
    return text


def translate_text(text, target_lang):
    if not text or not text.strip():
        raise gr.Error("There's no transcript to translate yet.")
    if target_lang == "None (keep original)":
        return text
    return _get_translator().translate(text, LANGUAGES[target_lang])


def make_files(text, formats, base_name):
    if not text or not text.strip():
        raise gr.Error("There's no text to export.")
    if not formats:
        raise gr.Error("Pick at least one output format.")
    base = (base_name or "transcript").strip() or "transcript"
    return [export_file(text, fmt, base) for fmt in formats]


# ---------------------------------------------------------------------------
# Real-time transcription handlers
# ---------------------------------------------------------------------------

def rt_process_stream(audio_chunk, state, model_name):
    """
    Called by Gradio every ~0.5 s while the microphone is active.

    audio_chunk : (sample_rate, np.ndarray) — new audio since last call
    state       : {"buffer": np.ndarray, "transcript": str}
    """
    if state is None:
        state = {"buffer": np.array([], dtype=np.float32), "transcript": ""}
    if audio_chunk is None:
        return state["transcript"], state

    sample_rate, data = audio_chunk
    if data.ndim > 1:
        data = data.mean(axis=1)  # stereo → mono

    buffer = np.concatenate([state["buffer"], data.astype(np.float32)])

    # Run Whisper only once we have CHUNK_SECONDS worth of audio
    if len(buffer) >= CHUNK_SECONDS * sample_rate:
        text = _get_rt_transcriber(model_name).transcribe_chunk(buffer, sample_rate)
        if text:
            sep = " " if state["transcript"] else ""
            state["transcript"] += sep + text
        buffer = np.array([], dtype=np.float32)

    state["buffer"] = buffer
    return state["transcript"], state


def rt_clear(state):
    state = {"buffer": np.array([], dtype=np.float32), "transcript": ""}
    return "", state


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="SpeechScribe Edge", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 🎙️ SpeechScribe Edge\n"
        "Transcribe audio with Whisper, optionally translate, edit, "
        "and export to PDF / DOCX / TXT / SRT."
    )

    with gr.Tabs():

        # ── Tab 1: File Transcription ─────────────────────────────────────
        with gr.Tab("File Transcription"):
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
                    chunk_sl = gr.Slider(
                        minimum=5,
                        maximum=60,
                        value=30,
                        step=5,
                        label="Chunk length (s) — audio window per Whisper call",
                    )
                    stride_sl = gr.Slider(
                        minimum=0,
                        maximum=10,
                        value=5,
                        step=1,
                        label="Stride length (s) — overlap on each side of chunk",
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
                        name_box = gr.Textbox(
                            label="File name (no extension)", value="transcript"
                        )
                    export_btn = gr.Button("3 · Generate files", variant="primary")
                    files_out = gr.File(label="Download", file_count="multiple")

            transcribe_btn.click(
                transcribe_audio,
                inputs=[audio_in, model_dd, ts_chk, chunk_sl, stride_sl],
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

        # ── Tab 2: Real-time Transcription ───────────────────────────────
        with gr.Tab("Real-time Transcription"):
            gr.Markdown(
                "Speak into your microphone — Whisper transcribes every "
                f"{CHUNK_SECONDS} seconds and appends to the transcript below. "
                "Use **tiny** or **base** for lowest latency."
            )

            rt_model_dd = gr.Dropdown(
                choices=[
                    "openai/whisper-tiny",
                    "openai/whisper-base",
                    "openai/whisper-small",
                ],
                value="openai/whisper-base",
                label="Whisper model",
            )

            with gr.Row():
                rt_mic = gr.Audio(
                    sources=["microphone"],
                    streaming=True,
                    type="numpy",
                    label="Microphone — click the mic and start speaking",
                )
                rt_transcript = gr.Textbox(
                    label="Live transcript (editable)",
                    lines=14,
                    placeholder="Start speaking — text appears here every few seconds.",
                    interactive=True,
                )

            rt_clear_btn = gr.Button("Clear transcript")
            rt_state = gr.State()

            rt_mic.stream(
                rt_process_stream,
                inputs=[rt_mic, rt_state, rt_model_dd],
                outputs=[rt_transcript, rt_state],
            )
            rt_clear_btn.click(
                rt_clear,
                inputs=[rt_state],
                outputs=[rt_transcript, rt_state],
            )


if __name__ == "__main__":
    demo.launch()
