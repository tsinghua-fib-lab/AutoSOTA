from enum import Enum, auto


class Setting(str, Enum):
    UNIVARIATE = "UNIVARIATE"
    MULTIVARIATE = "MULTIVARIATE"


class Prior(str, Enum):
    SN  = "SN"
    ISO = "ISO"
    OU  = "OU"
    SE  = "SE"
    PE  = "PE"


season_lengths = {
    "H": 24,
    "D": 30,
    "1D": 30,
    "B": 30,
    "const": 1,
}

season_lengths_gluonts = {
    "S": 3600,  # 1 hour
    "T": 1440,  # 1 day
    "H": 24,  # 1 day
    "1D": 1,
    "D": 1,  # 1 day
    "W": 1,  # 1 week
    "M": 12,
    "B": 5,
    "Q": 4,
    "Y": 1,
    "const": 1,
}


def get_season_length(freq):
    return season_lengths[freq]


def get_lags_for_freq(freq_str: str):
    if freq_str == "H":
        lags_seq = [24 * i for i in [1, 2, 3, 4, 5, 6, 7, 14, 21, 28]]
    elif freq_str == "B":
        # TODO: Fix lags for B
        lags_seq = [30 * i for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]
    elif freq_str == "1D":
        lags_seq = [30 * i for i in [1, 2, 3, 4, 5, 6, 7]]
    else:
        raise NotImplementedError(f"Lags for {freq_str} are not implemented yet.")
    return lags_seq
