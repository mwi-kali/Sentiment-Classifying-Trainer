import contractions
import emoji
import re
import string


from bs4 import BeautifulSoup
from .config import Settings
from transformers import AutoTokenizer
from typing import Any, Dict


class TextPreprocessor:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.tokenizer = AutoTokenizer.from_pretrained(settings.model_name)

    def is_allowed_char(self, ch):
        return ch.isprintable()
    
    def clean_text(self, text: str) -> str:
        text = emoji.demojize(text, delimiters=(' ', ' '))
        text = contractions.fix(text)
        text = BeautifulSoup(text, 'lxml').get_text(separator=' ')
        text = text.lower()
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#', '', text)
        keep = set('!?')
        remove = set(string.punctuation) - keep
        text = text.translate(str.maketrans('', '', ''.join(remove)))
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def tokenize(self, texts: list[str], clean: bool = False) -> Dict[str, Any]:
        if clean:
            texts = [self.clean_text(t) for t in texts]
        return self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.settings.max_length,
        )

    def prepare_dataset(self, dataset, clean: bool = False):
        def batch_tokenize(batch):
            texts = batch.get("text") or batch.get("sentence")
            if clean:
                texts = [self.clean_text(t) for t in texts]
            tokens = self.tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=self.settings.max_length,
            )
            return {
                "input_ids": tokens["input_ids"],
                "attention_mask": tokens["attention_mask"],
            }

        return dataset.map(batch_tokenize, batched=True)