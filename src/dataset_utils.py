from pathlib import Path

import polars as pl

DATASET_PATH = 'data/TRAIN_RELEASE_3SEP2025/train_subtask1.csv'
STATS_PATH = 'statistics/train_subtask1_group_by_users_len_sorted_asc.csv'

def prepare_df(dataset_path: str):
    df = pl.read_csv(dataset_path, try_parse_dates=True)
    return df
