import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
import os
import model_save
import nltk
from nltk.tokenize import sent_tokenize
import sentencepiece
import evaluate
from random import randrange

from transformers import DataCollatorForSeq2Seq, TrainingArguments, Trainer, pipeline, BitsAndBytesConfig
from transformers import LlamaTokenizer, LlamaForCausalLM
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from huggingface_hub import HfFolder
from trl import SFTTrainer

class Llama:
    def __init__(self, dataset_id, model_id, hyperparameters):
        user_id = model_save.get_user_id()
        model_save.insert_status_model(model_id,dataset_id)
        self.dataset_id = 'databricks/databricks-dolly-15k'
        self.model_id = model_id
        self.dataset = None
        self.tokenizer = None
        self.model = None
        self.data_collator = None
        self.training_args = hyperparameters
        self.trainer = None
        self.summarizer = None
        self.peft_config = None
        self.max_source_length = None
        self.max_target_length = None
        self.initialize()
        self.train_model()
        model_save.update_status_model(model_id, user_id)
        model_save.save_model(model_id, self.model, user_id)

    def initialize(self):
            # Load dataset from the hub
            self.dataset = load_dataset(self.dataset_id)

            self.dataset = self.dataset['train']

            # Load tokenizer
            self.tokenizer = LlamaTokenizer.from_pretrained(self.model_id)
            self.model = LlamaForCausalLM.from_pretrained(
                self.model_id,
                load_in_8bit=True,
                use_cache=False,
                use_flash_attention_2=False,
                device_map='auto',
                )

            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "right"

            # LoRA config based on QLoRA paper
            self.peft_config = LoraConfig(
                    lora_alpha=16,
                    lora_dropout=0.1,
                    r=64,
                    bias="none",
                    task_type="CAUSAL_LM",
            )

            self.model = prepare_model_for_kbit_training(self.model)


    def train_model(self):

      def format_prompt(sample):
        return f"""### Instruction:
        Use the Input below to create an instruction, which could have been used to generate the input using an LLM.

        ### Input:
        {sample['response']}

        ### Response:
        {sample['instruction']}
        """

      try:
        # Additional setup for training
        # model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id)
        repository_id = f"{self.model_id.split('/')[1]}-{self.dataset_id}"
        # Define training args
        training_args = TrainingArguments(
            output_dir="llama-7-int4-dolly",
            num_train_epochs=3,
            per_device_train_batch_size=3,
            gradient_accumulation_steps=2,
            gradient_checkpointing=True,
            optim="paged_adamw_32bit",
            logging_steps=10,
            save_strategy="epoch",
            learning_rate=2e-4,
            bf16=False,
            fp16=False,
            tf32=False,
            max_grad_norm=0.3,
            warmup_ratio=0.03,
            lr_scheduler_type="constant",
            warmup_steps=0,
            save_total_limit=3,
            disable_tqdm=False,  # disable tqdm since with packing values are in correct
            )

        model = get_peft_model(self.model, self.peft_config)

        # Create Trainer instance
        trainer = SFTTrainer(
            model=model,
            train_dataset = self.dataset,
            peft_config = self.peft_config,
            formatting_func = format_prompt,
            tokenizer=self.tokenizer,
            dataset_text_field="text",
            max_seq_length=2048,
            args=training_args,
            packing=True,
            )

        # Start training
        trainer.train()


      except Exception as e:
        print(f"Error during training: {str(e)}")

