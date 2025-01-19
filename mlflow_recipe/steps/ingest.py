
import logging

from pandas import DataFrame

_logger = logging.getLogger(__name__)


def load_file_as_dataframe(file_path: str, file_format: str) -> DataFrame:

    if file_format == "csv":
        import pandas
        df = pandas.read_csv(file_path, sep=";")
        df["is_red"] = 1 if "red" in str(file_path) else 0
        _logger.info(f"Loaded csv file from {file_path}")
        return df
    else:
        raise NotImplementedError
