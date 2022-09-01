"""Train a question answering model on the ScandiQA dataset."""

import hydra
from omegaconf import DictConfig

from scandi_qa.train import train_model


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(config: DictConfig):
    train_model(config)


if __name__ == "__main__":
    main()
