from functools import partial
from typing import Callable, Dict, Mapping, Sequence, Optional, Dict, Literal, Iterable, Hashable
from pathlib import Path
import pandas as pd
import numpy as np
import re
import os
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd

def explore_csv_hierarchy(root_dir, depth_name_list=None,allowed_ext=['.csv']):
    """
    Recursively explore a directory tree and list all CSV files
    along with their hierarchical structure.

    Parameters
    ----------
    root_dir : str
        Root directory to start the recursive search.
    depth_name_list : list[str] | None
        Optional list of column names for hierarchy levels.
        If None, levels are named as 'level_0', 'level_1', etc.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing one row per CSV file with:
        - one column per directory level,
        - 'filename' for the file name,
        - 'path' for the absolute file path.
    """
    data = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Filter out hidden directories (in-place to affect os.walk traversal)
        dirnames[:] = [d for d in dirnames if not d.startswith('.')]

        # Filter out hidden files
        filenames = [f for f in filenames if not f.startswith('.')]
        filenames = [f for f in filenames if "metadata" not in f]
        for file in filenames:
            if file.endswith(allowed_ext):
                full_path = os.path.join(dirpath, file)
                relative_path = os.path.relpath(full_path, root_dir)
                parts = relative_path.split(os.sep)
                
                if depth_name_list is None:
                    row = {f'level_{i}': part for i, part in enumerate(parts[:-1])}
                else:
                    row = {depth_name_list[i]: part for i, part in enumerate(parts[:-1])}
                row['filename'] = parts[-1]
                row['path'] = full_path
                data.append(row)
    df = pd.DataFrame(data)
    # Optional: sort columns (levels first, then filename/path)
    return df

def filter_metadata(metadata_df, constraint_selection_list=None, constraint_rejection_list=None):
    df_filtered = metadata_df.copy()

    # Apply inclusion constraints
    if constraint_selection_list:
        for col, allowed_values in constraint_selection_list:
            df_filtered = df_filtered[df_filtered[col].isin(allowed_values)]

    # Apply exclusion constraints
    if constraint_rejection_list:
        for col, rejected_values in constraint_rejection_list:
            df_filtered = df_filtered[~df_filtered[col].isin(rejected_values)]
    return df_filtered

def stratified_train_val_split(df, strat_cols, alpha=0.8, random_state=42):
    """
    Sépare un DataFrame en train et validation avec une stratification multi-colonnes.

    Paramètres
    ----------
    df : pd.DataFrame
        Le dataframe complet.
    strat_cols : list[str]
        Les colonnes utilisées pour la stratification (ex: ['col1', 'col2']).
    alpha : float
        Proportion d'exemples dans le train (entre 0 et 1).
    random_state : int
        Graine pour la reproductibilité.

    Retour
    ------
    df_train : pd.DataFrame
    df_val : pd.DataFrame
    """

    # On crée une colonne "strat_key" combinant les colonnes
    strat_key = df[strat_cols].astype(str).agg('_'.join, axis=1)

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        train_size=alpha,
        random_state=random_state
    )

    # split
    for train_idx, val_idx in splitter.split(df, strat_key):
        df_train = df.iloc[train_idx].copy()
        df_val = df.iloc[val_idx].copy()

    return df_train, df_val
