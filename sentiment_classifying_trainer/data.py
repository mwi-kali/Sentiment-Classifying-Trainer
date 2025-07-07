import logging

from .config import Settings
from datasets import load_dataset, DatasetDict


logger = logging.getLogger(__name__)


class SentimentDataLoader:
    def __init__(self, settings: Settings):
        self.settings = settings

    def load(self):
        ds_name = self.settings.dataset_name.lower()
        if ds_name == "tweet_eval":
            ds = load_dataset("tweet_eval", self.settings.dataset_subset)
        elif ds_name == "imdb":
            raw = load_dataset("imdb")
            split = raw["train"].train_test_split(test_size=0.1, seed=42)
            ds = DatasetDict({
                "train": split["train"], 
                "validation": split["test"], 
                "test": raw["test"]
            })
        elif ds_name == "financial_phrasebank":
            raw = load_dataset("financial_phrasebank", "sentences_allagree")
            split = raw["train"].train_test_split(test_size=0.1, seed=42)
            ds = DatasetDict({
                "train": split["train"],
                "validation": split["test"],
                "test": split["test"] 
            })
        else:
            raise ValueError(f"Unsupported dataset: {self.settings.dataset_name}")

        return ds
