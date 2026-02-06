#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import statsmodels.api as sm
from tqdm import tqdm


"""
yfinGenHelp.py
-----------
Core analytical tools for the Stock Database system.

This module contains:
    • Safety score regression fitting
    • Bootstrap coefficient estimation (10k sample default)
    • Vectorized confidence interval computation
    • Utility functions for applying CIs to full DataFrames

"""

# ============================================================
# 1. FIT SAFETY MODEL (4-factor regression)
# ============================================================

def fit_safety_model(df):
    """
    Fit the OLS safety model using the 4 standardized factors.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns:
            sharpe_1y_z
            vol_z
            downside_vol_z
            idiosyncratic_vol_z
            safety_score

    Returns
    -------
    model : statsmodels RegressionResults
    """

    required = [
        "sharpe_1y_z",
        "vol_z",
        "downside_vol_z",
        "idiosyncratic_vol_z",
        "safety_score"
    ]

    df_clean = df.dropna(subset=required).copy()

    X = df_clean[required[:-1]]  # all except safety_score
    X = sm.add_constant(X)

    y = df_clean["safety_score"]

    model = sm.OLS(y, X).fit()
    return model


# ============================================================
# 5. CHUNK LIST
# ============================================================

def chunk_list(lst, size=50):
    """Yield successive chunks of size `size`."""
    for i in range(0, len(lst), size):
        yield lst[i:i+size]


# ============================================================
# 6. ASSIGN ASSET TYPE
# ============================================================

def assign_asset_type(row):
    if row['ticker'] in STOCK_LIST:
        row['asset_type'] = "STOCK"
    elif row['ticker'] in MUTUAL_LIST:
        asset_type = "MUTUAL_FUND"
    elif row['ticker'] in ETF_LIST:
        asset_type = "ETF"
    else:
        asset_type = "UNKNOWN"

    row['asset_type'] = asset_type



"""
============================================================
=                   FUTURE EXPANSION                      =
============================================================

Add new functions below this block.

Examples you might add later:
    • ML training on your stock history table
    • Forecasting volatility or Sharpe ratios
    • Sector-level safety score models
    • Portfolio optimization functions

Steps When Adding New Functions:
--------------------------------
1. Write the new function inside this module.
2. Save this file: stockDB.py
3. In your VS Code script or notebook, import like:

       import stockDB
       from stockDB import my_new_function

4. If you're editing in Jupyter and the function does not update,
   run:

       from importlib import reload
       reload(stockDB)

This reloads the updated module without restarting the kernel.
============================================================
"""



# In[7]:


get_ipython().system('jupyter nbconvert --to python yfinGenHelp.ipynb')

