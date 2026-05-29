"""Whisper-based speech-to-text, GPU-aware."""

import torch
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline,
)


class Transcriber:
    def __init__(self, model_name: str = "openai/whisper-small"):
        self.model_name = model_name
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        model.to(self.device)
        processor = AutoProcessor.from_pretrained(model_name)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            batch_size=16,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )

    def transcribe(
        self,
        audio_path: str,
        return_timestamps: bool = False,
        chunk_length_s: int = 30,
        stride_length_s: int = 5,
    ) -> str:
        result = self.pipe(
            audio_path,
            return_timestamps=return_timestamps,
            chunk_length_s=chunk_length_s,
            stride_length_s=(stride_length_s, stride_length_s),
        )
        if return_timestamps and "chunks" in result:
            lines = []
            for ch in result["chunks"]:
                start, end = ch.get("timestamp", (None, None))
                stamp = f"[{_fmt(start)} - {_fmt(end)}] " if start is not None else ""
                lines.append(f"{stamp}{ch['text'].strip()}")
            return "\n".join(lines)
        return result["text"].strip()


def _fmt(seconds) -> str:
    if seconds is None:
        return "??:??"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
