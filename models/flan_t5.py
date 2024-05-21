from datasets import load_dataset
import pandas as pd
from random import randrange
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq
from transformers import pipeline
from datasets import concatenate_datasets
import evaluate
import nltk
import numpy as np
from nltk.tokenize import sent_tokenize
nltk.download("punkt")
from transformers import DataCollatorForSeq2Seq
from huggingface_hub import HfFolder
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import model_save


class FlanT5:
    def __init__(self, dataset_id, model_id, hyperparametes):
        user_id = model_save.get_user_id()
        self.dataset_id = dataset_id
        self.model_id = model_id
        self.dataset = None
        self.tokenizer = None
        self.model = None
        self.data_collator = None
        self.training_args = hyperparametes
        self.trainer = None
        self.summarizer = None
        self.max_source_length = None
        self.max_target_length = None
        self.tokenized_dataset = None
        self.initialize()
        self.train_model()
        model_save.save_model(model_id, self.model, dataset_id, user_id)

    def initialize(self):
            # Load dataset from the hub
            self.dataset = load_dataset(self.dataset_id)

            # Load tokenizer of FLAN-t5-base
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

            # The maximum total input sequence length after tokenization.
            tokenized_inputs = concatenate_datasets([self.dataset["train"], self.dataset["test"]]).map(
                lambda x: self.tokenizer(x["dialogue"], truncation=True),
                batched=True,
                remove_columns=["dialogue", "summary"]
            )
            self.max_source_length = max([len(x) for x in tokenized_inputs["input_ids"]])

            # The maximum total sequence length for target text after tokenization.
            tokenized_targets = concatenate_datasets([self.dataset["train"], self.dataset["test"]]).map(
                lambda x: self.tokenizer(x["summary"], truncation=True),
                batched=True,
                remove_columns=["dialogue", "summary"]
            )
            self.max_target_length = max([len(x) for x in tokenized_targets["input_ids"]])

            def preprocess_function(sample, padding="max_length"):
              # add prefix to the input for t5
              inputs = ["summarize: " + item for item in sample["dialogue"]]

              # tokenize inputs
              model_inputs = self.tokenizer(inputs, max_length=self.max_source_length, padding=padding, truncation=True)

              # Tokenize targets with the `text_target` keyword argument
              labels = self.tokenizer(text_target=sample["summary"], max_length=self.max_target_length, padding=padding, truncation=True)

              # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
              # padding in the loss.
              if padding == "max_length":
                  labels["input_ids"] = [
                      [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
                  ]

              model_inputs["labels"] = labels["input_ids"]
              return model_inputs

            self.tokenized_dataset = (self.dataset).map(preprocess_function, batched=True, remove_columns=["dialogue", "summary", "id"])

            # Initialize the tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

            # Model initialization
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id)

            # Data collator initialization
            label_pad_token_id = -100
            self.data_collator = DataCollatorForSeq2Seq(
                self.tokenizer,
                model=self.model,
                label_pad_token_id=label_pad_token_id,
                pad_to_multiple_of=8
            )


            # For example, initialize the summarizer pipeline
            #self.summarizer = pipeline("summarization", model="philschmid/flan-t5-base-samsum")

    def train_model(self):
      def show_metrics(self):
        # Metric
        metric = evaluate.load("rouge")
        return metric

      # helper function to postprocess text
      def postprocess_text(preds, labels):
          preds = [pred.strip() for pred in preds]
          labels = [label.strip() for label in labels]

          # rougeLSum expects newline after each sentence
          preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
          labels = ["\n".join(sent_tokenize(label)) for label in labels]

          return preds, labels

      def compute_metrics(self, eval_preds):
          preds, labels = eval_preds
          if isinstance(preds, tuple):
              preds = preds[0]
          decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
          # Replace -100 in the labels as we can't decode them.
          labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
          decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

          # Some simple post-processing
          decoded_preds, decoded_labels = self.postprocess_text(decoded_preds, decoded_labels)

          result = evaluate.load("rouge").compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
          result = {k: round(v * 100, 4) for k, v in result.items()}
          prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in preds]
          result["gen_len"] = np.mean(prediction_lens)
          return result

      try:

        # Additional setup for training
        # model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id)
        repository_id = f"{self.model_id.split('/')[1]}-{self.dataset_id}"
        # Define training args
        training_args = Seq2SeqTrainingArguments(
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            predict_with_generate=True,
            fp16=False, # Overflows with fp16
            learning_rate=5e-5,
            num_train_epochs=5,
            # logging & evaluation strategies
            #logging_dir=f"{repository_id}/logs",
            logging_strategy="steps",
            logging_steps=500,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            # metric_for_best_model="overall_f1",
            # push to hub parameters
            report_to="tensorboard",
            push_to_hub=False,
            hub_strategy="every_save",
            #hub_model_id=repository_id,
            hub_token=HfFolder.get_token(),
            output_dir=repository_id,
            )

        # Create Trainer instance
        trainer = Seq2SeqTrainer(
            model = self.model,
            args = training_args,
            data_collator = self.data_collator,
            train_dataset = self.tokenized_dataset["train"],
            eval_dataset = self.tokenized_dataset["test"],
            compute_metrics = compute_metrics,  # Make sure to define compute_metrics
        )

        # Start training
        trainer.train()

        # You might want to save the trained model or perform other post-training tasks here
      except Exception as e:
        print(f"Error during training: {str(e)}")


    def summarize_sample(self, sample):
            try:
                # Ensure that the summarizer is initialized
                if self.summarizer is None:
                    print("Summarizer is not initialized. Please ensure the model is trained.")
                    return None

                # Select a random test sample
                # You can use the provided sample instead of selecting a random one
                input_text = sample["dialogue"]

                # Summarize using the initialized summarizer
                summary_result = self.summarizer(input_text)

                # Extract the summary text from the result
                summary_text = summary_result[0]['summary_text']

                return summary_text

            except Exception as e:
                print(f"Error during summarization: {str(e)}")
                return None
