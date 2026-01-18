import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

sns.set_theme(style="whitegrid")

def plot_correlation_heatmap(df: pd.DataFrame):

    corr = df.corr()
    
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(
        corr, 
        mask=mask, 
        annot=True, 
        fmt=".2f", 
        cmap="coolwarm", 
        vmax=1, 
        vmin=-1,
        center=0,
        square=True,
        linewidths=.5,
        cbar_kws={"shrink": .5},
        ax=ax
    )
    
    ax.set_title("Mapa Korelacji Cech", fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    return fig

def plot_histogram(df: pd.DataFrame, column_name: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    
    sns.histplot(df[column_name], kde=True, ax=ax, bins=30)
    
    ax.set_title(f"Rozkład (Histogram) dla '{column_name}'", fontsize=14)
    ax.set_xlabel(column_name)
    ax.set_ylabel("Częstotliwość")
    
    return fig

def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray):

    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.scatter(y_true, y_pred, alpha=0.5, label="Dane Testowe")

    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  
        np.max([ax.get_xlim(), ax.get_ylim()]), 
    ]
    
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    
    ax.plot(lims, lims, 'r--', alpha=0.75, label="Predykcja (y=x)")
    
    ax.set_title("Rzeczywiste vs Przewidywane Wartości", fontsize=14)
    ax.set_xlabel("Rzeczywiste Wartości", fontsize=12)
    ax.set_ylabel("Przewidziane Wartości", fontsize=12)
    ax.legend()
    ax.grid(True)
    
    return fig
