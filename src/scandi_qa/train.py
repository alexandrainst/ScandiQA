"""Module related to training question answering models."""

import os
from typing import Dict

import hydra
from datasets import DownloadMode
from datasets.load import load_dataset, load_metric
from omegaconf import DictConfig
from transformers.data.data_collator import DefaultDataCollator
from transformers.models.auto.modeling_auto import AutoModelForQuestionAnswering
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.trainer import Trainer
from transformers.trainer_callback import EarlyStoppingCallback
from transformers.trainer_utils import EvalPrediction
from transformers.training_args import TrainingArguments


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def train_model(config: DictConfig) -> None:
    """Train a question answering model on ScandiQA.

    Args:
        config (DictConfig):
            Configuration object.
    """
    # Deal with full determinism
    if config.model.full_determinism:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = "4096:8"

    # Load the data
    dataset_dict = load_dataset(
        "alexandrainst/scandiqa",
        config.language,
        use_auth_token=True,
        download_mode=DownloadMode.FORCE_REDOWNLOAD,
    )

    # Create the tokeniser
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_id)

    # Create data collator
    data_collator = DefaultDataCollator()

    # Create the model
    model = AutoModelForQuestionAnswering.from_pretrained(
        config.model.model_id,
        use_auth_token=config.model.use_auth_token,
        from_flax=config.model.from_flax,
        cache_dir=".cache",
    )

    # Preprocess the data
    def preprocess_function(examples):
        inputs = tokenizer(
            examples["question"],
            examples["context"],
            max_length=384,
            truncation="only_second",
            return_offsets_mapping=True,
            padding=config.model.padding,
        )

        offsets = inputs.pop("offset_mapping")

        inputs["start_positions"] = list()
        inputs["end_positions"] = list()

        # We will label impossible answers with the index of the CLS token.
        input_ids = inputs["input_ids"]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the
        # context and what is the question).
        sequence_ids = inputs.sequence_ids()

        # Get the answers
        answers = examples["answers"]

        # If no answers are given, set the cls_index as answer.
        if len(answers["answer_start"]) == 0:
            inputs["start_positions"].append(cls_index)
            inputs["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature
            # is labeled with the CLS index).
            if not (
                offsets[token_start_index][0] <= start_char
                and offsets[token_end_index][1] >= end_char
            ):
                inputs["start_positions"].append(cls_index)
                inputs["end_positions"].append(cls_index)

            # Otherwise move the token_start_index and token_end_index to the two
            # ends of the answer. Note: we could go after the last offset if the
            # answer is the last word (edge case).
            else:
                while (
                    token_start_index < len(offsets)
                    and offsets[token_start_index][0] <= start_char
                ):
                    token_start_index += 1
                inputs["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                inputs["end_positions"].append(token_end_index + 1)

        return inputs

    # Apply the preprocessing
    dataset_dict = dataset_dict.map(preprocess_function)

    # Initialise the metrics
    em_metric = load_metric("exact_match")
    f1_metric = load_metric("f1")

    # Create the `compute_metrics` function
    def compute_metrics(predictions_and_labels: EvalPrediction) -> Dict[str, float]:
        """Compute the metrics for the transformer model.

        Args:
            predictions_and_labels (EvalPrediction):
                A tuple of predictions and labels.

        Returns:
            Dict[str, float]:
                The metrics.
        """
        # Extract the predictions
        predictions, labels = predictions_and_labels
        predictions = predictions.argmax(axis=-1)

        # Compute the metrics
        em = em_metric.compute(predictions=predictions, references=labels)[
            "exact_match"
        ]
        f1 = f1_metric.compute(predictions=predictions, references=labels)["f1"]

        return dict(em=em, f1=f1)

    # Create early stopping callback
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=config.model.early_stopping_patience
    )

    # Set up output directory
    output_dir = f"{config.models.dir}/{config.model.name}"

    # Create the training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy=config.model.evaluation_strategy,
        logging_strategy=config.model.logging_strategy,
        save_strategy=config.model.save_strategy,
        eval_steps=config.model.eval_steps,
        logging_steps=config.model.logging_steps,
        save_steps=config.model.save_steps,
        max_steps=config.model.max_steps,
        report_to=config.model.report_to,
        save_total_limit=config.model.save_total_limit,
        per_device_train_batch_size=config.model.batch_size,
        per_device_eval_batch_size=config.model.batch_size,
        learning_rate=config.model.learning_rate,
        warmup_ratio=config.model.warmup_ratio,
        gradient_accumulation_steps=config.model.gradient_accumulation_steps,
        optim=config.model.optim,
        seed=config.seed,
        full_determinism=config.model.full_determinism,
        lr_scheduler_type=config.model.lr_scheduler_type,
        fp16=config.model.fp16,
        metric_for_best_model="f1",
        greater_is_better=True,
        load_best_model_at_end=True,
    )

    # Create the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["val"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping_callback],
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    print(trainer.evaluate(dataset_dict["test"]))

    # Save the model
    trainer.save_model()

    # Return the model
    return model


if __name__ == "__main__":
    train_model()
