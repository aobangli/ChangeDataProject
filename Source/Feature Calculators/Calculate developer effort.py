import pandas as pd
import numpy as np
from Source.Config import *
import joblib


def main():
    change_list_path = f'{root}/{project}_change_list.csv'

    change_list_df = joblib.load(change_list_path)
    selected_change_list = joblib.load(f"{root}/{project}_selected_change_list.csv")
    change_list_df = change_list_df[change_list_df['change_id'].isin(selected_change_list['change_id'])]
    print_details(change_list_df, cols=['duration', 'message_num', 'revision_num'])


def filter_anomalies(df: pd.DataFrame, col: str):
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    # Create a mask in between q1 & q3
    mask = df[col].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr, inclusive=True)

    # Filtering the initial dataframe with a mask
    return df.loc[mask]


def print_details(df: pd.DataFrame, cols: list[str]):
    print('Columns, Total, Merged, Abandoned')
    for col in cols:
        df_filtered = filter_anomalies(df, col)
        mean_total = np.mean(df_filtered[col])
        mean_merged = np.mean(df_filtered[df_filtered['status'] == 'MERGED'][col])
        mean_abandoned = np.mean(df_filtered[df_filtered['status'] != 'MERGED'][col])

        print(f'{col}, {mean_total}, {mean_merged}, {mean_abandoned}')
    print()


if __name__ == '__main__':
    main()