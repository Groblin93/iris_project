import pickle

import hydra
import mlflow
import onnx
import pytorch_lightning as pl
import torch
from iris_project.my_types import data_module, lightning_nn_classification_model
from mlflow import MlflowClient
from mlflow.models import infer_signature


@hydra.main(config_path="configs", config_name="config", version_base="1.3.2")
def main(cfg):
    pl.seed_everything(42)

    dm = data_module(
        path=cfg.data.path,
        name=cfg.data.name,
        X_name=cfg.data.X_name,
        y_name=cfg.data.y_name,
        repo=cfg.data.repo,
        col_names=cfg.data.col_names,
        nums=cfg.data.nums,
        test_size=cfg.data.test_size,
        val_size=cfg.data.val_size,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
    )

    model = lightning_nn_classification_model(cfg)

    mlf_logger = pl.loggers.MLFlowLogger(
        experiment_name=cfg.logg.exp_name,
        tracking_uri=cfg.logg.uri,
    )
    mlf_logger.log_hyperparams(
        params={"git commit id": "58d34d9d1ceb8b9b851aaad475e967ba82f8cc46"}
    )

    callbacks = [
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
        pl.callbacks.DeviceStatsMonitor(),
        pl.callbacks.RichModelSummary(max_depth=cfg.callbacks.model_summary.max_depth),
    ]

    trainer = pl.Trainer(
        max_epochs=cfg.training.num_epochs,
        log_every_n_steps=cfg.training.log_step,
        logger=mlf_logger,
        callbacks=callbacks,
    )

    trainer.fit(model, train_dataloaders=dm)

    client = MlflowClient(tracking_uri=cfg.logg.uri)
    client.search_experiments()

    pickle.dump(model, open(cfg.model.path + cfg.model.name, "wb"))

    X_input = torch.randn(20, 4)
    torch.onnx.export(model, X_input, cfg.model.path + cfg.model.name_onnx)
    onnx_model = onnx.load_model(cfg.model.path + cfg.model.name_onnx)
    mlflow.set_tracking_uri(cfg.logg.uri)
    with mlflow.start_run():
        signature = infer_signature(X_input.numpy(), model(X_input).detach().numpy())
        mlflow.onnx.log_model(onnx_model, "Iris_model", signature=signature)
    # onnx_pyfunc = mlflow.pyfunc.load_model(model_info.model_uri)

    # predictions = onnx_pyfunc.predict(X_input.numpy())
    # print(predictions)


if __name__ == "__main__":
    main()
