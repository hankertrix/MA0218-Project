import marimo

__generated_with = "0.11.22"
app = marimo.App(width="full", app_title="MA2018 Mini Project")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # MA2018 Mini Project
        This project makes use of the climate change dataset.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Import all the required libraries.""")
    return


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sb

    # Set the seaborn style
    sb.set_theme()
    return mo, np, pd, plt, sb


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The constants used in the program.""")
    return


@app.cell
def _():
    DATA_FILE = "./data.xls"
    return (DATA_FILE,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Read the data from the data file.""")
    return


@app.cell
def _(DATA_FILE, pd):
    data = pd.read_excel(DATA_FILE)
    print(data.head())
    return (data,)


if __name__ == "__main__":
    app.run()
