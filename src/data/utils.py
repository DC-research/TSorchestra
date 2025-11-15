from collections.abc import Iterable, Iterator

from gluonts.dataset import DataEntry
from gluonts.dataset.field_names import FieldName
from gluonts.ev.metrics import (
    MAE,
    MAPE,
    MASE,
    MSE,
    MSIS,
    ND,
    NRMSE,
    RMSE,
    SMAPE,
    BaseMetricDefinition,
    MeanWeightedSumQuantileLoss,
)
from gluonts.transform import Transformation

QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

M4_PRED_LENGTH_MAP = {
    "H": 48,  # Hourly
    "h": 48,
    "D": 14,  # Daily
    "d": 14,
    "W": 13,  # Weekly
    "w": 13,
    "M": 18,  # Monthly
    "m": 18,
    "ME": 18,  # End of month
    "Q": 8,  # Quarterly
    "q": 8,
    "QE": 8,  # End of quarter
    "A": 6,  # Annualy/yearly
    "y": 6,
    "YE": 6,  # End of year
}

PRED_LENGTH_MAP = {
    "S": 60,  # Seconds
    "s": 60,
    "T": 48,  # Minutely
    "min": 48,
    "H": 48,  # Hourly
    "h": 48,
    "D": 30,  # Daily
    "d": 30,
    "W": 8,  # Weekly
    "w": 8,
    "M": 12,  # Monthly
    "m": 12,
    "ME": 12,
    "Q": 8,  # Quarterly
    "q": 8,
    "QE": 8,
    "y": 6,  # Annualy/yearly
    "A": 6,
}

# Prediction lengths from the TFB benchmark: https://arxiv.org/abs/2403.20150
TFB_PRED_LENGTH_MAP = {
    "U": 8,
    "T": 8,  # Minutely
    "H": 48,  # Hourly
    "h": 48,
    "D": 14,  # Daily
    "W": 13,  # Weekly
    "M": 18,  # Monthly
    "Q": 8,  # Quarterly
    "A": 6,  # Annualy/yearly
}


def itemize_start(data_entry: DataEntry) -> DataEntry:
    """
    Converts the `start` field into a native Python type.
    """
    data_entry[FieldName.START] = data_entry[FieldName.START].item()
    return data_entry


def get_metrics() -> list[BaseMetricDefinition]:
    return [
        MSE(forecast_type="mean"),
        MSE(forecast_type=0.5),
        MAE(),
        MASE(),
        MAPE(),
        SMAPE(),
        MSIS(),
        RMSE(),
        NRMSE(),
        ND(),
        MeanWeightedSumQuantileLoss(quantile_levels=QUANTILE_LEVELS),
    ]


class MultivariateToUnivariate(Transformation):
    """
    Unpacks a single `D` dimensional multivariate time series into `D`
    separate univariate time series.
    """

    def __init__(self, field: str = FieldName.TARGET):
        self.field = field

    def __call__(
        self,
        dataset: Iterable[DataEntry],
        is_train: bool = False,
    ) -> Iterator:
        """
        Converts a multivariate dataset into univariate by unpacking each
        dimension into a separate entry.

        Args:
            dataset (Iterable[DataEntry]): The dataset to convert from
                multivariate to univariate format.
            is_train (bool, optional): Whether the transformation is being used
                during training (not used in this case). Defaults to False.
            NOTE: Keep `is_train=False` to maintain compatibility with GluonTS.


        Yields:
            Iterator: An iterator over the univariate entries, where each entry
                has the same fields as the original dataset, but with the
                target field containing only one dimension of the original
                multivariate target, and the item_id field modified to reflect
                the dimension.
        """
        for data_entry in dataset:
            item_id = data_entry[FieldName.ITEM_ID]
            multivariate_target = list(data_entry[self.field])
            for id, univariate_target in enumerate(multivariate_target):
                univariate_entry = data_entry.copy()
                univariate_entry[self.field] = univariate_target
                univariate_entry[FieldName.ITEM_ID] = f"{item_id}_dim{id}"
                yield univariate_entry
