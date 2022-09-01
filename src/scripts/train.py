"""Train a question answering model on the ScandiQA dataset."""

from hydra import compose, initialize

from scandi_qa.train import train_model

# Initialise Hydra
initialize(config_path="../../config", version_base=None)


def main():
    """Train a question answering model on the ScandiQA dataset."""

    # Build config
    config = compose("config")

    # Train the model
    train_model(config)


if __name__ == "__main__":
    main()
