from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Dict, Optional


class Settings(BaseSettings):
    dataset_name: str = Field("tweet_eval", description="HuggingFace dataset")
    dataset_subset: Optional[str] = Field("sentiment", description="Subset for tweet_eval")
    epochs: int = Field(3, description="Number of training epochs")
    eval_batch_size: int = Field(16)
    financial_dataset_name: str = Field("financial_phrasebank", description="HuggingFace dataset")
    financial_output_dir: str = Field("./trained_models/financial_models", description="Checkpoint directory")
    id2label: Dict[int, str] = Field({0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}, description="Label mapping")
    learning_rate: float = Field(2e-5)
    max_length: int = Field(128, description="Max token length")
    model_name: str = Field("distilbert-base-uncased", description="HuggingFace model")
    num_labels: int = Field(3, description="Number of sentiment classes")
    output_dir: str = Field("./trained_models/general_models", description="Checkpoint directory")
    train_batch_size: int = Field(16)
