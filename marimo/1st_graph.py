import marimo

__generated_with = "0.26.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    df = pd.read_csv("data/resultats_simple_1000.csv")
    df
    return df, np, plt


@app.cell
def _(df):
    # Range
    df["decennie"] = (df["annee"] // 10) * 10

    # Color frequent to rare
    ordre_couleurs = df["couleur"].value_counts().index

    # Nb of cars per 10 years and color (0 if none)
    table = df.groupby(["decennie", "couleur"]).size().unstack(fill_value=0)

    # Change the counts to %
    table = table.div(table.sum(axis=1), axis=0) * 100
    table = table[ordre_couleurs].sort_index()
    table
    return (table,)


@app.cell
def _(np, plt, table):
    couleurs = {
        "white": "#f2f2f2", "black": "#222222", "silver": "#c0c0c0",
        "gray": "#808080", "anthracite": "#4a4f55", "red": "#d62728",
        "blue": "#1f5fbf", "green": "#2ca02c", "turquoise": "#279c90",
        "beige": "#d8c8a0", "brown": "#8b5a2b", "yellow": "#f5d90a",
        "orange": "#ff8c00", "bordeaux": "#6d0f1f", "pink": "#ff8fb1",
        "gold": "#d4a017", "violet": "#7f3fbf"
    }

    colors = [couleurs[c] for c in table.columns]

    x = np.arange(len(table))

    fig, ax = plt.subplots()

    # Add data to graph
    ax.stackplot(
        x,
        table.T,
        labels=table.columns,
        colors=colors
    )

    # Modif graph line
    ax.set_xticks(x)
    ax.set_xticklabels(table.index.astype(str))
    ax.set_xlim(0, len(table) - 1)
    return ax, fig


@app.cell
def _(ax):
    # Grid
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    ax.grid(axis="y", linestyle=":", alpha=0.2)


@app.cell
def _(ax, fig):
    # Graph
    ax.set_title("Distribution des couleurs de voitures par décennie")
    ax.set_xlabel("Décennie")
    ax.set_ylabel("Pourcentage des couleurs")

    ax.set_ylim(0, 100)

    ax.legend(title="Couleur", bbox_to_anchor=(1.02, 1))

    fig


if __name__ == "__main__":
    app.run()
