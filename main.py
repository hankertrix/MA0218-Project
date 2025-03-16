# The Python file in the percent Jupyter notebook format.
#
# Format specification:
# https://jupytext.readthedocs.io/en/latest/formats-scripts.html

# %% [markdown]
# # MA2018 Mini Project
# This project makes use of the climate change dataset.

# %% [markdown]
# Import all the required libraries.

# %%
import pandas as pd
import numpy as np

# %% [markdown]
# The constants used in the program.

# %%
DATA_FILE = "./data.xls"

# %% [markdown]
# Read the data from the data file.

# %%
data = pd.read_excel(DATA_FILE)
