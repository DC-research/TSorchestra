import logging
from pathlib import Path
import json
import pandas as pd
from gluonts.model import evaluate_model

from src.data.dataset import Dataset
from src.data.utils import get_metrics
from src.models.common.gluonts_predictor import GluonTSPredictor
from src.utils.path import resolve_dataset_properties_path, resolve_output_path

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Evaluator for GIFT-Eval datasets.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        verbose: bool = True,
    ) -> None:
        """
        Initialize an Evaluator instance for a specific dataset.

        Args:
            dataset (Dataset): The dataset to evaluate on.
            batch_size (int): The batch size to use for evaluation.
            verbose (bool): Whether to print progress information.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.verbose = verbose
        
    @property
    def dataset_properties_map(self) -> dict:
        dataset_properties_path = resolve_dataset_properties_path()
        with open(dataset_properties_path, "r") as f:
            return json.load(f)
        
    @property
    def ds_key(self) -> str:
        pretty_names = {
            "saugeenday": "saugeen",
            "temperature_rain_with_missing": "temperature_rain",
            "kdd_cup_2018_with_missing": "kdd_cup_2018",
            "car_parts_with_missing": "car_parts",
        }
        if "/" in self.dataset.name:
            ds_key = self.dataset.name.split("/").iloc[0]
            ds_key = ds_key.lower()
        else:
            ds_key = self.dataset.name.lower()
        return pretty_names.get(ds_key, ds_key)

    def evaluate(self, predictor: GluonTSPredictor) -> None:
        """
        Evaluate a GluonTS predictor on a dataset and save results.

        NOTE: Follow the conventions outlined in the GIFT-Eval repo to remain
        compatible with their leaderboard. See here for more details:
        https://github.com/SalesforceAIResearch/gift-eval?tab=readme-ov-file#evaluation

        Args:
            predictor (GluonTSPredictor): The predictor to evaluate.
        """
        res = evaluate_model(
            predictor,
            test_data=self.dataset.test_data,
            metrics=get_metrics(),
            batch_size=self.batch_size,
            axis=None,
            mask_invalid_label=True,
            allow_nan_forecast=False,
            seasonality=self.dataset.seasonality,
        )

        # Prepare the results for the CSV file
        results_data = [
            [
                self.dataset.config,
                predictor.alias,
                res["MSE[mean]"].iloc[0],
                res["MSE[0.5]"].iloc[0],
                res["MAE[0.5]"].iloc[0],
                res["MASE[0.5]"].iloc[0],
                res["MAPE[0.5]"].iloc[0],
                res["sMAPE[0.5]"].iloc[0],
                res["MSIS"].iloc[0],
                res["RMSE[mean]"].iloc[0],
                res["NRMSE[mean]"].iloc[0],
                res["ND[0.5]"].iloc[0],
                res["mean_weighted_sum_quantile_loss"].iloc[0],
                self.dataset_properties_map[self.ds_key]["domain"],
                self.dataset_properties_map[self.ds_key]["num_variates"],
            ]
        ]

        if self.verbose:
            mase = res["MASE[0.5]"].iloc[0]
            crps = res["mean_weighted_sum_quantile_loss"].iloc[0]
            logging.info(f"MASE: {mase:.4f}, CRPS: {crps:.4f}")

        results_df = pd.DataFrame(
            results_data,
            columns=[
                "dataset",
                "model",
                "eval_metrics/MSE[mean]",
                "eval_metrics/MSE[0.5]",
                "eval_metrics/MAE[0.5]",
                "eval_metrics/MASE[0.5]",
                "eval_metrics/MAPE[0.5]",
                "eval_metrics/sMAPE[0.5]",
                "eval_metrics/MSIS",
                "eval_metrics/RMSE[mean]",
                "eval_metrics/NRMSE[mean]",
                "eval_metrics/ND[0.5]",
                "eval_metrics/mean_weighted_sum_quantile_loss",
                "domain",
                "num_variates",
            ],
        )
        
        output_path = resolve_output_path(alias=predictor.alias, dataset_config=self.dataset.config)
        csv_file_path = output_path / "results.csv"
        csv_file_path.parent.mkdir(parents=True, exist_ok=True)
        if csv_file_path.exists():
            results_df = pd.concat(
                [pd.read_csv(csv_file_path), results_df],
                ignore_index=True,
            )
        results_df.to_csv(csv_file_path, index=False)
