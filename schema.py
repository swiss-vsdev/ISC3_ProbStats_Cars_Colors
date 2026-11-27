import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    import marimo as mo
    import numpy as np
    import seaborn as sb
    import matplotlib.pyplot as plt
    import plotly.express as px
    from sklearn.model_selection import train_test_split

    df = pd.read_csv("data/train.csv").set_index("PassengerId")
    df
    return df, mo, np, pd, plt, train_test_split


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Question 1
    ## a. Identifiez les valeurs manquantes.
    """)
    return
