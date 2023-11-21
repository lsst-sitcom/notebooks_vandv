import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from pathlib import Path

from lsst.ts.xml.tables.m1m3 import HP_COUNT


def merge_csvs(folder_name: str | Path, file_pattern: str, list_of_day_obs: list):
    """
    Merge the csvs files associated with the dayObs inside ``day_obs_list``

    Parameters
    ----------
    folder_name : str or Path
        Path containing the CSVs files.

    file_pattern : str
        The pattern of the filename with a placeholder for ``day_obs`` values.

    list_of_day_obs : list
        List containing the relevant dayObs.

    Returns
    -------
    df : pd.DataFrame
        Merged Pandas Dataframes.
    """
    # Create a Path instance for the target directory
    path = Path(folder_name)

    # Find files that match the pattern for each day_obs value
    matching_files = []
    for day_obs in list_of_day_obs:
        pattern = file_pattern.format(day_obs=day_obs)
        matching_files.extend(path.glob(pattern))

    df = pd.DataFrame()
    for file in matching_files:
        temp_df = pd.read_csv(file)
        df = pd.concat((df, temp_df))

    return df


def correlation_map(df: pd.DataFrame, columns_to_use: list, lines: list | None = None) -> None:
    """
    Display a correlation map for ``df`` while using only the columns in
    ``columns_to_use``.

    Parameters
    ----------
    df : pd.DataFrame
        M1M3 ICS Summary data
    columns_to_use : list of str
        Names of the columns that we use for the correlation plot.
    lines : list of ints, optional
        If provided, add a line after each N-th variable. 
    """
    filtered_df = df[columns_to_use]
    corr = filtered_df.corr()

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    fig = plt.figure(figsize=(20, 15))
    ax = sns.heatmap(
        corr,
        cmap=cmap,
        vmax=0.3,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
    )
    
    if lines:
        ax.hlines(lines, *ax.get_xlim(), colors="black", alpha=0.5)
        ax.vlines(lines, *ax.get_ylim(), colors="black", alpha=0.5)
    
    plt.title("Further Updated Correlation Map")
    plt.show()
    

def multiaxis_plots(df: pd.DataFrame, xcol: str, ycol_prefix: str):
    """
    Create a multi-axes plot. One axes for each hard-point data.

    Parameters
    ----------
    df : pd.DataFrame
        M1M3 ICS Summary data
    xcol : str
        Name of the column to use for the X-Axis.
    ycol_prefix : str
        Prefix of the columns that will be used as Y-Axis on each subplot.
    """
    fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(15, 30), sharex=True)

    colors = {1: "blue", -1: "red"}
    ycols = [f"{ycol_prefix}{hp}" for hp in range(HP_COUNT)]

    for i, ycol in enumerate(ycols):
        axes[i].scatter(df[xcol], df[ycol], c=df["ics_enabled"].map(colors), alpha=0.5)
        axes[i].set_ylabel(ycol)
        axes[i].grid(True)

        # Adding legend for 'ics_enabled' values
        for ics_val, color in colors.items():
            axes[i].scatter([], [], color=color, label=f"ics_enabled={ics_val}")

        axes[i].legend()

    axes[-1].set_xlabel(f"{xcol}")
    plt.suptitle(f"{ycol_prefix} vs {xcol}", y=1.02)
    plt.tight_layout()
    plt.show()


def singleaxis_plots(df: pd.DataFrame, xcol: str, ycol_prefix: str):
    """
    Create a single-axes plot. One color for each hard-point data.

    Parameters
    ----------
    df : pd.DataFrame
        M1M3 ICS Summary data
    xcol : str
        Name of the column to use for the X-Axis.
    ycol_prefix : str
        Prefix of the columns that will be used as Y-Axis on each subplot.
    """
    fig, ax = plt.subplots(figsize=(15, 10))

    alpha = {1: 1.0, -1: 0.25}
    ycols = [f"{ycol_prefix}{hp}" for hp in range(HP_COUNT)]

    for i, ycol in enumerate(ycols):
        ax.scatter(df[xcol], df[ycol], alpha=df["ics_enabled"].map(alpha), label=ycol)
        ax.set_ylabel(ycol)
        ax.grid(True)

        # # Adding legend for 'ics_enabled' values
        # for ics_val, color in colors.items():
        #     axes[i].scatter([], [], color=color, label=f"ics_enabled={ics_val}")

    ax.legend()
    ax.set_xlabel(f"{xcol}")
    plt.suptitle(f"{ycol_prefix} vs {xcol}", y=1.02)
    plt.tight_layout()
    plt.show()
