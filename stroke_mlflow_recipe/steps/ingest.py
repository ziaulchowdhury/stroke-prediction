
import logging
from pandas import DataFrame

def load_file_as_dataframe(file_path: str, file_format: str) -> DataFrame:
    _logger = logging.getLogger(__name__)

    if file_format == "csv":
        import pandas
        df = pandas.read_csv(file_path,)
        _logger.info(f"Loaded csv file from {file_path}")
        return df
    else:
        raise NotImplementedError
