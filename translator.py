"""NLLB-based translation. Add or remove languages in LANGUAGES as needed."""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "facebook/nllb-200-distilled-600M"

# Friendly name -> NLLB language code (FLORES-200).
LANGUAGES = {
    "Arabic": "arb_Arab",
    "Chinese (Simplified)": "zho_Hans",
    "English": "eng_Latn",
    "French": "fra_Latn",
    "German": "deu_Latn",
    "Hindi": "hin_Deva",
    "Italian": "ita_Latn",
    "Japanese": "jpn_Jpan",
    "Korean": "kor_Hang",
    "Portuguese": "por_Latn",
    "Russian": "rus_Cyrl",
    "Spanish": "spa_Latn",
    "Turkish": "tur_Latn",
    "Urdu": "urd_Arab",
}


class Translator:
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(self.device)

    def translate(self, text: str, target_code: str) -> str:
        # Translate line-by-line so structure/timestamps survive.
        out_lines = []
        for line in text.split("\n"):
            if not line.strip():
                out_lines.append("")
                continue
            inputs = self.tokenizer(line, return_tensors="pt", truncation=True).to(self.device)
            bos = self.tokenizer.convert_tokens_to_ids(target_code)
            tokens = self.model.generate(
                **inputs, forced_bos_token_id=bos, max_length=512
            )
            out_lines.append(
                self.tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]
            )
        return "\n".join(out_lines)
