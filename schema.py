import marimo

__generated_with = "0.26.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    import marimo as mo
    import numpy as np
    import seaborn as sb
    import seaborn.objects as so
    import matplotlib.pyplot as plt

    df = pd.read_csv("resultats_simple_1000.csv")
    df
    return (df,)


@app.cell
def _(df):
    df["decennie"] = (df["annee"] // 10) * 10

    # Nb of cars per 10 years and color (0 if none)
    table = df.groupby(["decennie", "couleur"]).size().unstack(fill_value=0)

    # Change the counts to %
    table = table.div(table.sum(axis=1), axis=0)

    # Long format for seaborn
    # .reset_index() to get the decade
    # .melt() merge the column to two
    data = table.reset_index().melt(id_vars="decennie", var_name="couleur", value_name="part")
    data


@app.cell
def _():
    couleurs = {
        "white": "#f2f2f2", "black": "#222222", "silver": "#c0c0c0",
        "gray": "#808080", "anthracite": "#4a4f55", "red": "#d62728",
        "blue": "#1f5fbf", "green": "#2ca02c", "turquoise": "#40e0d0",
        "beige": "#d8c8a0", "brown": "#8b5a2b", "yellow": "#f5d90a",
        "orange": "#ff8c00", "bordeaux": "#6d0f1f", "pink": "#ff8fb1",
        "gold": "#d4a017", "violet": "#7f3fbf",
    }


if __name__ == "__main__":
    app.run()
