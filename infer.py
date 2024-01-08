import pickle

import hydra
import pandas as pd
import pytorch_lightning as pl
from iris_project.my_types import data_set
from torch.utils.data import DataLoader


@hydra.main(config_path="configs", config_name="config", version_base="1.3.2")
def main(cfg):
    model = pickle.load(open(cfg.model.path + cfg.model.name, "rb"))
    test_X = pd.read_csv(cfg.data.path + cfg.data.X_name, delimiter=",")
    test_y = pd.read_csv(cfg.data.path + cfg.data.y_name, delimiter=",")
    test_dataset = data_set(test_X.values, test_y.values[:, 0])
    trainer = pl.Trainer()
    trainer.test(
        model, dataloaders=DataLoader(test_dataset, num_workers=cfg.data.num_workers)
    )


if __name__ == "__main__":
    main()
