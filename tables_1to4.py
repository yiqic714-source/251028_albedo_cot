"""
tables_1to4.py

Print sensitivity_albedo_vs_cot_1030.csv and sensitivity_albedo_vs_cot_day.csv
in a formatted table for Tables 1-4.

For each CSV, print:
  (k_msk +/- k_msk_unc, lnb_msk +/- lnb_msk_unc)
  (k_ret +/- k_ret_unc, lnb_ret +/- lnb_ret_unc)

Format: first row is season names, first column of each row is ocean name.
"""

import os
import numpy as np
import pandas as pd

BASE_PATH = '/home/chenyiqi/251028_albedo_cot'
PROCESSED_DIR = os.path.join(BASE_PATH, 'processed_data')

CSV_FILES = [
    'sensitivity_albedo_vs_cot_1030.csv',
    'sensitivity_albedo_vs_cot_day.csv',
]

SEASONS = ['MAM', 'JJA', 'SON', 'DJF']
OCEANS = ['NPO', 'NAO', 'TPO', 'TAO', 'TIO', 'SPO', 'SAO', 'SIO']

# Column name mapping for different CSV files
# _1030.csv uses: k_ret, lnb_ret, k_ret_unc, lnb_ret_unc, k_msk, lnb_msk, k_msk_unc, lnb_msk_unc
# _day.csv uses:  k_ret_day, lnb_ret_day, k_ret_day_unc, lnb_ret_day_unc, k_msk_day, lnb_msk_day, k_msk_day_unc, lnb_msk_day_unc
CSV_COLUMNS = {
    'sensitivity_albedo_vs_cot_1030.csv': {
        'k_ret': 'k_ret', 'lnb_ret': 'lnb_ret', 'k_ret_unc': 'k_ret_unc', 'lnb_ret_unc': 'lnb_ret_unc',
        'k_msk': 'k_msk', 'lnb_msk': 'lnb_msk', 'k_msk_unc': 'k_msk_unc', 'lnb_msk_unc': 'lnb_msk_unc',
    },
    'sensitivity_albedo_vs_cot_day.csv': {
        'k_ret': 'k_ret_day', 'lnb_ret': 'lnb_ret_day', 'k_ret_unc': 'k_ret_day_unc', 'lnb_ret_unc': 'lnb_ret_day_unc',
        'k_msk': 'k_msk_day', 'lnb_msk': 'lnb_msk_day', 'k_msk_unc': 'k_msk_day_unc', 'lnb_msk_unc': 'lnb_msk_day_unc',
    },
}


def fmt_val(val, unc):
    """Format value and uncertainty as 'val+/-unc' with 3 decimal places."""
    if np.isnan(val) or np.isnan(unc):
        return 'NaN'
    return f'{val:.3f}+/-{unc:.3f}'


def print_table(csv_name, label_k, label_lnb, k_col, k_unc_col, lnb_col, lnb_unc_col):
    """
    Print a formatted table from a CSV file.

    Parameters
    ----------
    csv_name : str
        Name of the CSV file (e.g., 'sensitivity_albedo_vs_cot_1030.csv').
    label_k : str
        Label for the k column (e.g., 'k_msk').
    label_lnb : str
        Label for the lnb column (e.g., 'lnb_msk').
    k_col : str
        Column name for k values.
    k_unc_col : str
        Column name for k uncertainty.
    lnb_col : str
        Column name for lnb values.
    lnb_unc_col : str
        Column name for lnb uncertainty.
    """
    csv_path = os.path.join(PROCESSED_DIR, csv_name)
    if not os.path.exists(csv_path):
        print(f'\n{"=" * 70}')
        print(f'File not found: {csv_path}')
        print(f'{"=" * 70}')
        return

    df = pd.read_csv(csv_path)

    print(f'\n{"=" * 70}')
    print(f'Table: {csv_name}  |  Variable: ({label_k}, {label_lnb})')
    print(f'{"=" * 70}')

    # Header row: first column is "Ocean", then seasons
    header = f'{"Ocean":<8s}'
    for s in SEASONS:
        header += f'  {s:>20s}'
    print(header)
    print('-' * len(header))

    for ocean in OCEANS:
        row_str = f'{ocean:<8s}'
        for season in SEASONS:
            mask = (df['Ocean'] == ocean) & (df['Season'] == season)
            if mask.any():
                row = df[mask].iloc[0]
                k_val = row[k_col]
                k_unc = row[k_unc_col]
                lnb_val = row[lnb_col]
                lnb_unc = row[lnb_unc_col]
                cell = f'{fmt_val(k_val, k_unc)}, {fmt_val(lnb_val, lnb_unc)}'
            else:
                cell = 'N/A'
            row_str += f'  {cell:>20s}'
        print(row_str)

    print()


def main():
    for csv_name in CSV_FILES:
        col_map = CSV_COLUMNS[csv_name]

        # Print k_msk +/- k_msk_unc, lnb_msk +/- lnb_msk_unc
        print_table(
            csv_name=csv_name,
            label_k='k_msk',
            label_lnb='lnb_msk',
            k_col=col_map['k_msk'],
            k_unc_col=col_map['k_msk_unc'],
            lnb_col=col_map['lnb_msk'],
            lnb_unc_col=col_map['lnb_msk_unc'],
        )

        # Print k_ret +/- k_ret_unc, lnb_ret +/- lnb_ret_unc
        print_table(
            csv_name=csv_name,
            label_k='k_ret',
            label_lnb='lnb_ret',
            k_col=col_map['k_ret'],
            k_unc_col=col_map['k_ret_unc'],
            lnb_col=col_map['lnb_ret'],
            lnb_unc_col=col_map['lnb_ret_unc'],
        )


if __name__ == '__main__':
    main()
