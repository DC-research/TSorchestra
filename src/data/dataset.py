import json
import math
from functools import cached_property
from typing import Literal

from datasets import Dataset as HFDataset
from datasets import load_from_disk
from datasets.utils.logging import disable_progress_bar
from dotenv import load_dotenv
from dotted_dict import DottedDict
from gluonts.dataset.common import ProcessDataEntry
from gluonts.dataset.split import TestData, TrainingDataset, split
from gluonts.itertools import Map
from gluonts.time_feature import get_seasonality, norm_freq_str
from pandas.tseries.frequencies import to_offset
from toolz import compose

from src.data.utils import (
    M4_PRED_LENGTH_MAP,
    PRED_LENGTH_MAP,
    MultivariateToUnivariate,
    itemize_start,
)
from src.utils.enums import Domain, Term
from src.utils.path import (
    resolve_metadata_path,
    resolve_storage_path,
)


class Dataset:
    def __init__(
        self,
        name: str,
        term: Term | Literal["short", "medium", "long"] = Term.SHORT,
        verbose: bool = False,
        storage_env_var: str = "GIFT_EVAL",
        **kwargs,
    ):
        """
        Class for loading GIFT-Eval datasets and their metadata.

        This class was based on GIFT-Eval's `Dataset` class. See here for the
        original implementation:
        https://github.com/SalesforceAIResearch/gift-eval/blob/main/src/gift_eval/data.py

        Args:
            name (str): Name of the dataset to load.
            term (Term | str): Forecast horizon term. Used to optionally scale
                the dataset's original prediction length.
            verbose (bool): Whether to display a progress bar when loading the
                dataset from disk. Defaults to False.
            storage_env_var (str): Environment variable pointing to the stored
                datasets' root directory.
            **kwargs: Additional keyword arguments, if any.
        """
        self.name = name
        self.term = Term(term)
        self.storage_env_var = storage_env_var

        if not verbose:
            disable_progress_bar()

        # Set format to numpy before creating gluonts dataset
        self.hf_dataset.set_format("numpy")

        process_data_entry = ProcessDataEntry(
            self.freq,
            one_dim_target=self.target_dim == 1,
        )

        self.gluonts_dataset = Map(
            compose(process_data_entry, itemize_start),
            self.hf_dataset,
        )

        # Automatically convert multivariate datasets to univariate
        if self.target_dim > 1:
            self.gluonts_dataset = MultivariateToUnivariate().apply(
                self.gluonts_dataset
            )

        # Handle additional keyword arguments
        for key, value in kwargs.items():
            setattr(self, key, value)

    @cached_property
    def hf_dataset(self) -> HFDataset:
        """
        Loads the underlying Hugging Face dataset from disk.
        """
        load_dotenv()
        storage_path = resolve_storage_path(
            storage_env_var=self.storage_env_var,
        )
        return load_from_disk(storage_path / self.name)

    @cached_property
    def metadata(self) -> DottedDict:
        """
        Loads the dataset's metadata from a JSON file.

        Returns:
            DottedDict: A DottedDict containing the dataset's metadata. Allows
                key names to be accessed using dot notation.
        """
        with open(resolve_metadata_path()) as f:
            metadata = json.load(f)
        return DottedDict(metadata[self.name])

    @property
    def domain(self) -> Domain:
        """
        Returns the dataset's domain.
        """
        return Domain(self.metadata.domain)

    @property
    def config(self) -> str:
        """
        Returns the dataset's configuration formatted as `name`/`freq`/`term`.
        This's is used for formatting dataset names and terms in results files.
        - `name` is the dataset's name in lowercase, with some additional
            formatting for specific datasets.
            - E.g. "saugeenday" is formatted as "saugeen".
        - `freq` is the dataset's frequency with the optional dash removed.
            - E.g. "W-SUN" is formatted as "W".
        - `term` is the dataset's term (short, medium, long).

        Returns:
            str: The dataset's configuration formatted as `name`/`freq`/`term`.
        """
        pretty_names = {
            "saugeenday": "saugeen",
            "temperature_rain_with_missing": "temperature_rain",
            "kdd_cup_2018_with_missing": "kdd_cup_2018",
            "car_parts_with_missing": "car_parts",
        }
        name = self.name.split("/")[0] if "/" in self.name else self.name
        cleaned_name = pretty_names.get(name.lower(), name.lower())
        cleaned_freq = self.freq.split("-")[0]
        return f"{cleaned_name}/{cleaned_freq}/{self.term}"

    @property
    def seasonality(self) -> int:
        """
        Computes the dataset's seasonality (number of time steps in one
        seasonal cycle). This's a thin wrapper around GluonTS's
        `get_seasonality`.

        Returns:
            int: The dataset's seasonality.
        """
        return get_seasonality(self.freq)

    @property
    def freq(self) -> str:
        """
        Returns the dataset's frequency.
        """
        return self.metadata.freq

    @property
    def base_freq(self) -> str:
        """
        Returns the dataset's base frequency, which is the frequency without
        any offsets (e.g., "5T" becomes "T", "W-SUN" becomes "W").
        """
        return norm_freq_str(to_offset(self.freq).name)

    @property
    def target_dim(self) -> int:
        """
        Returns the number of dimensions in the dataset's target. Assumes all
        series in dataset have the same target dimension.
        """
        return self.metadata.target_dim

    @property
    def sum_series_length(self) -> int:
        """
        Returns the total number of observations across all series in the
        dataset.
        """
        return self.metadata.sum_series_length

    @property
    def _min_series_length(self) -> int:
        """
        Returns the minimum series length across all series in the dataset.
        """
        return self.metadata._min_series_length

    @property
    def _total_univariate_series(self) -> int:
        """
        Returns the total number of univariate series in the dataset after
        conversion from multivariate to univariate format.

        **NOTE**: This is different from `num_series`, which returns the total
        number of (potentially multivariate) series before converting the
        dataset from multivariate to univariate format.
        """
        return self.metadata._total_univariate_series

    @property
    def context_length(self) -> int:
        """
        Returns the dataset's context length determined by the term.
        """
        return self.term.context_length

    @cached_property
    def prediction_length(self) -> int:
        pred_len = (
            M4_PRED_LENGTH_MAP[self.base_freq]
            if "m4" in self.name
            else PRED_LENGTH_MAP[self.base_freq]
        )
        return self.term.multiplier * pred_len

    @cached_property
    def windows(self) -> int:
        if "m4" in self.name:
            return 1
        w = math.ceil(
            self.test_split * self._min_series_length / self.prediction_length
        )
        return min(max(1, w), self.max_windows)

    @property
    def training_offset(self) -> int:
        """
        Returns the number of observations to exclude from the end of each
        series when creating the training split.
        """
        return self.prediction_length * (self.windows + 1)

    @property
    def val_test_offset(self) -> int:
        """
        Returns the number of observations to exclude from the end of each
        series when creating the validation split. For the test split, this's
        the number of observations to use at the end of each series.
        """
        return self.prediction_length * self.windows

    @property
    def training_dataset(self) -> TrainingDataset:
        training_dataset, _ = split(
            self.gluonts_dataset,
            offset=-self.training_offset,
        )
        return training_dataset

    @property
    def validation_dataset(self) -> TrainingDataset:
        validation_dataset, _ = split(
            self.gluonts_dataset,
            offset=-self.val_test_offset,
        )
        return validation_dataset

    @property
    def test_data(self) -> TestData:
        _, test_template = split(
            self.gluonts_dataset,
            offset=-self.val_test_offset,
        )
        test_data = test_template.generate_instances(
            prediction_length=self.prediction_length,
            windows=self.windows,
            distance=self.prediction_length,
        )
        return test_data

    @property
    def m4_pred_length_map(self) -> dict[str, int]:
        """
        Returns the M4 prediction length map.
        """
        return {
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

    @property
    def pred_length_map(self) -> dict[str, int]:
        """
        Returns the prediction length map.
        """
        return {
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

    @property
    def tfb_pred_length_map(self) -> dict[str, int]:
        """
        Prediction lengths from the TFB benchmark:
        https://arxiv.org/abs/2403.20150
        """
        return {
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
