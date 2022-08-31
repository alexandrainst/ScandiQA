"""Module related to training question answering models."""

import os

import hydra
from datasets import DownloadMode
from datasets.load import load_dataset
from omegaconf import DictConfig
from transformers.data.data_collator import default_data_collator
from transformers.models.auto.modeling_auto import AutoModelForQuestionAnswering
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.trainer import Trainer
from transformers.trainer_callback import EarlyStoppingCallback
from transformers.training_args import TrainingArguments


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def train_model(config: DictConfig) -> None:
    """Train a question answering model on ScandiQA.

    Args:
        config (DictConfig):
            Configuration object.
    """
    # Deal with full determinism
    if config.training.full_determinism:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = "4096:8"

    # Load the data
    dataset_dict = load_dataset(
        "alexandrainst/scandiqa",
        config.language,
        use_auth_token=True,
        download_mode=DownloadMode.FORCE_REDOWNLOAD,
        cache_dir=".cache",
    )

    # Create the tokeniser
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_id)

    # Create the model
    model = AutoModelForQuestionAnswering.from_pretrained(
        config.model.model_id,
        use_auth_token=config.model.use_auth_token,
        from_flax=config.model.from_flax,
        cache_dir=".cache",
    )

    # Store whether the tokenizer pads on the right
    pad_on_right = tokenizer.padding_side == "right"

    # Preprocess the data
    def prepare_features(examples):

        # Some of the questions have lots of whitespace on the left, which is not
        # useful and will make the truncation of the context fail (the tokenized
        # question will take a lots of space). So we remove that left whitespace
        examples["question"] = [q.lstrip() for q in examples["question"]]

        # Tokenize our examples with truncation and padding, but keep the overflows
        # using a stride. This results in one example possible giving several features
        # when a context is long, each of those features having a context that overlaps
        # a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples["question" if pad_on_right else "context"],
            examples["context" if pad_on_right else "question"],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=config.model.max_length,
            stride=config.model.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        # Since one example might give us several features if it has a long context, we
        # need a map from a feature to its corresponding example. This key gives us
        # just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # The offset mappings will give us a map from token to character position in
        # the original context. This will help us compute the start_positions and
        # end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # Let's label those examples!
        tokenized_examples["start_positions"] = list()
        tokenized_examples["end_positions"] = list()

        for i, offsets in enumerate(offset_mapping):

            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the
            # context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example
            # containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples["answers"][sample_index]

            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)

            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature
                # is labeled with the CLS index).
                if not (
                    offsets[token_start_index][0] <= start_char
                    and offsets[token_end_index][1] >= end_char
                ):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)

                # Otherwise move the token_start_index and token_end_index to the two
                # ends of the answer. Note: we could go after the last offset if the
                # answer is the last word (edge case).
                else:
                    while (
                        token_start_index < len(offsets)
                        and offsets[token_start_index][0] <= start_char
                    ):
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

    # Apply the preprocessing to the training dataset
    dataset_dict = dataset_dict.map(
        prepare_features,
        batched=True,
        remove_columns=dataset_dict["train"].column_names,
    )

    # Create early stopping callback
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=config.training.early_stopping_patience
    )

    # Set up output directory
    output_dir = f"{config.models_dir}/{config.model.name}"

    # Create the training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy=config.training.evaluation_strategy,
        logging_strategy=config.training.logging_strategy,
        save_strategy=config.training.save_strategy,
        eval_steps=config.training.eval_steps,
        logging_steps=config.training.logging_steps,
        save_steps=config.training.save_steps,
        max_steps=config.training.max_steps,
        report_to=config.training.report_to,
        save_total_limit=config.training.save_total_limit,
        per_device_train_batch_size=config.training.batch_size,
        per_device_eval_batch_size=config.training.batch_size,
        learning_rate=config.training.learning_rate,
        warmup_ratio=config.training.warmup_ratio,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        optim=config.training.optim,
        seed=config.seed,
        full_determinism=config.training.full_determinism,
        lr_scheduler_type=config.training.lr_scheduler_type,
        fp16=config.training.fp16,
        metric_for_best_model="loss",
        greater_is_better=False,
        load_best_model_at_end=True,
        use_mps_device=config.training.use_mps_device,
    )

    # Create the trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=default_data_collator,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["val"],
        callbacks=[early_stopping_callback],
    )

    # Train the model
    trainer.train()

    # Save the model
    trainer.save_model()

    # Return the model
    return model


if __name__ == "__main__":
    train_model()
