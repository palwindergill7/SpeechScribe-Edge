"""Real-time speech-to-text using Whisper on buffered microphone chunks."""

import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

DEFAULT_MODEL = "openai/whisper-base"
SAMPLE_RATE = 16_000   # Whisper always expects 16 kHz
CHUNK_SECONDS = 3      # seconds to accumulate before each inference call


class RealtimeTranscriber:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name,
            torch_dtype=dtype,
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
            torch_dtype=dtype,
            device=self.device,
        )

    def transcribe_chunk(self, audio: np.ndarray, sample_rate: int) -> str:
        """Transcribe a numpy audio chunk. Returns stripped text or ''."""
        if audio is None or len(audio) == 0:
            return ""
        if sample_rate != SAMPLE_RATE:
            import librosa
            audio = librosa.resample(
                audio.astype(np.float32), orig_sr=sample_rate, target_sr=SAMPLE_RATE
            )
        audio = audio.astype(np.float32)
        if audio.max() > 1.0:  # int16 PCM → normalize to [-1, 1]
            audio /= 32768.0
        result = self.pipe({"array": audio, "sampling_rate": SAMPLE_RATE})
        return result["text"].strip()
