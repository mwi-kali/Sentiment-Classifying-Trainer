import logging

from .config import Settings
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from .utils import compute_metrics


logger = logging.getLogger(__name__)


class SentimentTrainer:
    def __init__(self, settings: Settings, tokenizer):
        self.settings = settings
        self.tokenizer = tokenizer
        self.model = None
        self.trainer = None

    def setup(self, train_ds, eval_ds):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.settings.model_name, 
            num_labels=self.settings.num_labels,
            id2label=self.settings.id2label,
            label2id={v:k for k,v in self.settings.id2label.items()}
        )
        args = TrainingArguments(
            output_dir=self.settings.output_dir,
            num_train_epochs=self.settings.epochs,
            per_device_train_batch_size=self.settings.train_batch_size,
            per_device_eval_batch_size=self.settings.eval_batch_size,
            learning_rate=self.settings.learning_rate,
            do_train=True,
            do_eval=True,
            eval_steps=500,   
            save_steps=500,      
            save_total_limit=2,  
            logging_dir="./logs",
            logging_steps=50,
        )
        self.trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics,
        )

        logger.info("Trainer initialized")

    def train(self):
        self.trainer.train()

    def evaluate(self, ds):
        raw_metrics = self.trainer.evaluate(ds)
        cleaned = {}
        for key, value in raw_metrics.items():
            if key.startswith("eval_"):
                cleaned[key[len("eval_"):]] = value
            else:
                cleaned[key] = value
        return cleaned

    def predict(self, ds):
        return self.trainer.predict(ds)
