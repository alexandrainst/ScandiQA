"""Module related to training question answering models."""

import os
from typing import Dict

import hydra
from datasets import load_dataset, load_metric
from omegaconf import DictConfig
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DefaultDataCollator,
    EarlyStoppingCallback,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)


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
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        answers = examples["answers"]["text"]
        answer_starts = examples["answers"]["answer_start"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            answer = answers[i]
            start_char = answer_starts[0]
            end_char = start_char + len(answer)
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label it (0, 0)
            if (
                offset[context_start][0] > end_char
                or offset[context_end][1] < start_char
            ):
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

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
