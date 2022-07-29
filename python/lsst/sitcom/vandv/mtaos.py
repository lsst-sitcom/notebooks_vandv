def get_dof_indexes(component):
    """Returns a dictionary containing the indexes used to unravel
    the Degrees of Freedom into M2Hex, CamHex, M1M3, and M2.

    Parameters
    ----------
    component : str
        One of the following: [m2hex, camhex, m1m3, m2]

    Returns
    -------
    dict : dictionary containing the start index and end index of
    each component.

    See Also
    --------
    Information extracted from:
      - https://ls.st/Document-14771 - Active Optics Baseline Design, Table 1.1
    """
    # M2Hex Dof
    #   Piston, x-decenter, y-decenter, rotX, rotY
    if component.lower() == "m2hex":
        idxs = dict(
            startIdx=0,
            idxLength=5,
            state0name="M2Hexapod",
        )
    # Camhex Dof
    #   Piston, x-decenter, y-decenter, rotX, rotY
    elif component.lower() == "camhex":
        idxs = dict(
            startIdx=5,
            idxLength=5,
            state0name="cameraHexapod",
        )

    # M1M3
    #   Bending modes represented as Zernike Coefficients
    #   from Z4 to Z21
    elif component.lower() == "m1m3":
        idxs = dict(startIdx=10, idxLength=20, state0name="M1M3Bending", rot_mat=1.0)
    # M2
    #   Zernike Coefficients from Z4 to Z21
    elif component.lower() == "m2":
        idxs = dict(startIdx=30, idxLength=20, state0name="M2Bending", rot_mat=1.0)
    else:
        raise ValueError(
            f"Expected `component` to be one of: [m2hex, camhex, m1m3, m2]."
            f" Found '{component}'"
        )

    return idxs


def show_dof(ax, df, component, symbols=None, labels=None):
    """Displays the M2Hex Degrees of Freedom.

    Parameters
    ----------
    ax : matplotlib.pyplot.Axes
        Axis used to plot the data
    df : pd.DataFrame
        DataFrame containing the Degrees of Freedom (aggregated or visit).
        Each row is displayed as a new series.
    component : str
        Name of the component. Any in ["m1m3", "m2", "camhex", "m2hex"].
    symbols : list of str
        Matplotlib representation of color+symbol per series (e.g. "ko-")
    labels : list of str
        Labels applied to each series.

    Returns
    ------
    ax : the axes updated with the plot for further customization.
    """

    # Check inputs
    if symbols:
        if len(symbols) != len(df):
            raise ValueError(
                f"Expected the number of elements in `symbols` ({len(symbols)})"
                f"to be the same as the number of rows in `df` ({len(df)})"
            )
    else:
        symbols = ["-"] * len(df)

    if labels:
        if len(labels) != len(df):
            raise ValueError(
                f"Expected the number of elements in `labels` ({len(labels)})"
                f"to be the same as the number of rows in `df` ({len(df)})"
            )
    else:
        labels = [""] * len(df)

    if component.lower() in ["m2hex", "camhex"]:
        new_cols = ["piston", "xDecenter", "yDecenter", "xRotation", "yRotation"]
    elif component.lower() in ["m1m3", "m2"]:
        new_cols = [f"z{i+4}" for i in range(20)]

    # Triage data from DataFrame
    idxs = get_dof_indexes(component)
    cols = df.columns[idxs["startIdx"] : idxs["startIdx"] + idxs["idxLength"]]

    _df = df[cols].copy()
    _df.columns = new_cols

    # Plot the data
    for i in range(len(_df)):
        ax.plot(_df.iloc[i], symbols[i], label=labels[i])

    ax.legend()
    ax.set_xlabel(f"{component}\nDoF")
    ax.grid(":", alpha=0.25)
    # ax.title = "foo"
