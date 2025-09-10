#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training module for Intrextro AGI models.
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

import torch
from datasets import Dataset
from transformers import (AutoModelForCausalLM,AutoTokenizer,DataCollatorForLanguageModeling,Trainer,TrainingArguments)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    model_name: str = "intrextro-base"
    learning_rate: float = 3e-5
    epochs: int = 3
    batch_size: int = 4
    max_length: int = 2048
    train_data_path: str = "data/training/default_train.json"
    valid_data_path: Optional[str] = None
    checkpoint_dir: str = "models/checkpoints"
    output_dir: str = "models/trained"
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05


class IntrextroTrainer:
    """Handles training and fine-tuning of Intrextro AGI models."""

    def __init__(self, config: TrainingConfig):
        """
        Initialize the trainer with configuration.

        Args:
                config: TrainingConfig object
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.output_dir, exist_ok=True)

    def load_model(self) -> None:
        """Load the model and tokenizer."""
        try:
            logger.info(f"Loading model: {self.config.model_name}")

            # Determine if we're loading from local path or HF hub
            if os.path.exists(self.config.model_name):
                model_path = self.config.model_name
            else:
                model_path = self.config.model_name

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            if not self.tokenizer.pad_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model with appropriate quantization for training
            if self.config.use_lora:
                from peft import LoraConfig, get_peft_model

                # Load base model with quantization
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    load_in_8bit=True,
                    device_map="auto",
                    torch_dtype=torch.float16,
                )

                # Apply LoRA configuration
                lora_config = LoraConfig(
                    r=self.config.lora_rank,
                    lora_alpha=self.config.lora_alpha,
                    lora_dropout=self.config.lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM",
                )

                # Create PEFT model
                self.model = get_peft_model(self.model, lora_config)
                self.model.print_trainable_parameters()
            else:
                # Standard loading without quantization for full fine-tuning
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path, device_map="auto"
                )

            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def _prepare_dataset(self, data_path: str) -> Dataset:
        """
        Prepare dataset from conversation JSON file.

        Args:
                data_path: Path to training data JSON

        Returns:
                HF Dataset object
        """
        try:
            with open(data_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            conversations = data.get("conversations", [])

            # Process conversations into training format
            training_samples = []
            for i in range(0, len(conversations), 2):
                if i + 1 < len(conversations):
                    user_msg = conversations[i].get("content", "")
                    assistant_msg = conversations[i + 1].get("content", "")

                    # Format as instruction
                    full_text = f"<|user|>\n{user_msg}\n<|assistant|>\n{assistant_msg}<|endoftext|>"
                    training_samples.append({"text": full_text})

            # Create dataset
            dataset = Dataset.from_list(training_samples)

            # Tokenize dataset
            def tokenize_function(examples):
                return self.tokenizer(
                    examples["text"],
                    padding="max_length",
                    truncation=True,
                    max_length=self.config.max_length,
                )

            tokenized_dataset = dataset.map(
                tokenize_function, batched=True, remove_columns=["text"]
            )

            return tokenized_dataset

        except Exception as e:
            logger.error(f"Error preparing dataset: {e}")
            raise

    def train_with_config(
        self, training_config: Optional[TrainingConfig] = None
    ) -> None:
        """
        Train model using provided or default configuration.

        Args:
                training_config: Optional TrainingConfig to override default
        """
        if training_config:
            self.config = training_config

        # Load model if not already loaded
        if self.model is None or self.tokenizer is None:
            self.load_model()

        try:
            # Prepare datasets
            logger.info(f"Preparing training data from {self.config.train_data_path}")
            train_dataset = self._prepare_dataset(self.config.train_data_path)

            valid_dataset = None
            if self.config.valid_data_path:
                valid_dataset = self._prepare_dataset(self.config.valid_data_path)

            # Set up training arguments
            training_args = TrainingArguments(
                output_dir=self.config.checkpoint_dir,
                num_train_epochs=self.config.epochs,
                per_device_train_batch_size=self.config.batch_size,
                per_device_eval_batch_size=self.config.batch_size,
                gradient_accumulation_steps=4,
                evaluation_strategy="epoch" if valid_dataset else "no",
                save_strategy="epoch",
                logging_dir=os.path.join(self.config.checkpoint_dir, "logs"),
                logging_steps=10,
                learning_rate=self.config.learning_rate,
                weight_decay=0.01,
                warmup_ratio=0.03,
                save_total_limit=3,
                load_best_model_at_end=True if valid_dataset else False,
                report_to="none",
            )

            # Create data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer, mlm=False
            )

            # Initialize trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=valid_dataset,
                data_collator=data_collator,
            )

            # Start training
            logger.info("Starting training")
            trainer.train()

            # Save trained model
            output_path = os.path.join(
                self.config.output_dir,
                f"{self.config.model_name.split ( '/' ) [-1]}_finetuned_{trainer.state.global_step}",
            )
            trainer.save_model(output_path)
            self.tokenizer.save_pretrained(output_path)

            logger.info(f"Training complete. Model saved to {output_path}")

        except Exception as e:
            logger.error(f"Training error: {e}")
            raise
