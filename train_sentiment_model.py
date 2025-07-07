import argparse
import logging

from datasets import DatasetDict

from sentiment_classifying_trainer.config import Settings
from sentiment_classifying_trainer.data import SentimentDataLoader
from sentiment_classifying_trainer.preprocess import TextPreprocessor
from sentiment_classifying_trainer.trainer import SentimentTrainer
from sentiment_classifying_trainer.utils import log_confusion_matrix


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser("Train and Evaluate Sentiment Model")
    parser.add_argument("--clean", action="store_true", help="Apply text cleaning")
    args = parser.parse_args()

    settings = Settings()
    loader = SentimentDataLoader(settings)
    pre = TextPreprocessor(settings)
    trainer = SentimentTrainer(settings, pre.tokenizer)

    ds: DatasetDict = loader.load()

    for split in ds.keys():
        ds[split] = ds[split].map(
            lambda ex: {"text": ex.get("text", ex.get("sentence", ""))},
            remove_columns=[c for c in ds[split].column_names if c not in ("text","label")]
        )
        ds[split] = pre.prepare_dataset(ds[split], clean=args.clean)
        ds[split].set_format("torch", columns=["input_ids", "attention_mask", "label"])

    trainer.setup(ds["train"], ds["validation"])
    trainer.train()

    metrics = trainer.evaluate(ds["test"])
    logger.info(f"Test metrics {metrics}")

    preds = trainer.predict(ds["test"])
    y_pred = preds.predictions.argmax(axis=1)
    y_true = preds.label_ids
    log_confusion_matrix(y_pred, y_true)


if __name__ == "__main__":
    main()
