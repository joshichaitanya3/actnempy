import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Generating a nematic field with a pair of defects

    In this first example, we will generate a nematic field with a pair of
    $\pm 1/2$ defects, plot the nematic director and find the defect positions and
    orientations.

    ### Modules/functions used:
    * nematic_plot
    * defect_finder
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np

    from actnempy.utils import defect_finder as df
    from actnempy.utils import nematic_plot

    return df, nematic_plot, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Function to generate the nematic director
    """)
    return


@app.cell
def _(np):
    def defectpair(X, Y, dist, varphi1, varphi2, rcore=0.1):
        """
        Function to generate the orientation profile for a pair
        of +-1/2 defects centered at (-dist/2,0) and (dist/2,0) respectively.
        varphi1 and varphi2 set the orientation of the minus and the plus half defects.
        rcore sets the defect core size.
        Refer to Eq. (33) of X. Tang and J. V. Selinger, Soft Matter 13, 5481 (2017).
        """
        dth = varphi2 - varphi1 + np.pi / 2
        Th = varphi1 - np.pi / 2
        th = (
            -0.5 * np.arctan2(X + 0.5 * dist, Y)
            + 0.5 * np.arctan2(X - 0.5 * dist, Y)
            + 0.5
            * dth
            * (
                1
                + (np.log((X + 0.5 * dist) ** 2 + Y**2) - np.log((X - 0.5 * dist) ** 2 + Y**2))
                / (2 * np.log(dist / rcore))
            )
            + Th
        )

        th[((X + 0.5 * dist) ** 2 + Y**2 < rcore**2)] = 0
        th[((X - 0.5 * dist) ** 2 + Y**2 < rcore**2)] = 0

        return th.T

    return (defectpair,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Generate and plot the director
    """)
    return


@app.cell
def _(mo):
    dist = mo.ui.slider(1, 10, label="Distance between defects", value=10)
    return (dist,)


@app.cell
def _(defectpair, dist, np):
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)

    dx = x[1] - x[0]
    X, Y = np.meshgrid(x, y, indexing="ij")

    th = defectpair(X, Y, dist.value, np.pi / 3, 0)
    return dx, th, x, y


@app.cell
def _(dist):
    dist
    return


@app.cell
def _(nematic_plot, np, plt, th, x, y):
    nx = np.cos(th)
    ny = np.sin(th)

    nematic_plot(x, y, nx, ny, density=2)
    ax1 = plt.gca()
    ax1.set_aspect("equal")
    ax1
    return nx, ny


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Identify defect positions and orientations
    """)
    return


@app.cell
def _(mo):
    filter_radius = mo.ui.slider(1, 5, label="Filter Radius", value=5)
    area_threshold = mo.ui.slider(0, 100, label="Area threshold for defect position", value=60)
    return area_threshold, filter_radius


@app.cell
def _(area_threshold, df, filter_radius, nx, ny):
    # create charge density map
    _, map_p, map_m = df.func_defectfind(nx, ny, filter_radius=filter_radius.value, switchsign=0)

    # search map and identify circular regions of positive and negative charge
    centroids_p = df.func_defectpos(map_p, areathresh=area_threshold.value)

    centroids_m = df.func_defectpos(map_m, areathresh=area_threshold.value)

    # get the oriengation of defects
    phi_p = df.func_defectorient(
        centroids_p, nx, ny, filter_radius=filter_radius.value, type_str="positive"
    )
    phi_m = df.func_defectorient(
        centroids_m, nx, ny, filter_radius=filter_radius.value, type_str="negative"
    )
    return centroids_m, centroids_p, phi_m, phi_p


@app.cell
def _(
    centroids_m,
    centroids_p,
    df,
    dx,
    nematic_plot,
    nx,
    ny,
    phi_m,
    phi_p,
    plt,
    x,
    y,
):
    fig, ax = plt.subplots()
    nematic_plot(x, y, nx, ny, density=2.0)
    ax.set_aspect("equal", adjustable="box")
    color_p = "magenta"
    color_m = "cyan"
    defect_scale = 1

    cp = centroids_p * dx - 10
    cm = centroids_m * dx - 10
    df.func_plotdefects(ax, cp, phi_p, color_p, "positive", defect_scale)
    df.func_plotdefects(ax, cm, phi_m, color_m, "negative", defect_scale)

    plt.xlabel(r"x")
    plt.ylabel(r"y")
    return


@app.cell
def _(area_threshold, dist, filter_radius):
    dist, filter_radius, area_threshold
    return


if __name__ == "__main__":
    app.run()
