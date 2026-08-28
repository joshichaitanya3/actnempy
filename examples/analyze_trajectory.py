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
    # Importing a trajectory of an active nematic

    In this example, we import the data of a trajectory of an active nematic, which
    includes the $Q_{xx}$, $Q_{xy}$, $u_x$ and $u_y$ values at all points in space and
    time.

    ### Analysis includes:

    * Visualizing the director, scalar order, flow field and vorticity
    * Computing the velocity and orientation auto-correlation function in time
    * Computing the flow-field divergence for a single frame, as well as averaging
      over all time to check incompressibility
    * Finding the defect positions and orientations in a given frame, plotting the
      result, and computing the total number of $\pm 1/2$ defects over time.
    """)
    return


@app.cell
def _():
    import gdown
    import matplotlib.pyplot as plt

    from actnempy import ActNem

    return ActNem, gdown, plt


@app.cell
def _(gdown, mo):
    url = "https://drive.google.com/uc?id=1BYS1iVh9rCR_aNSnPk2qodJzDuh_2mI6"

    data_dir = mo.notebook_dir().parent / "TestData"
    output = data_dir / "processed_data.npz"

    # Only fetch the dataset if it isn't already on disk, so that re-running the
    # notebook (or a reactive re-run of this cell) doesn't re-download it.
    if not output.exists():
        gdown.download(url, str(output), quiet=False)
    return data_dir, output


@app.cell
def _(ActNem, data_dir):
    an = ActNem(data_dir)
    return (an,)


@app.cell
def _(an):
    an.visualize(1, save=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Compute the velocity autocorrelation function in time
    """)
    return


@app.cell
def _(an):
    (vcorr, _tc) = an.velocity_autocorr()
    print(f'Velocity autocorrelation time: {_tc} units')
    return (vcorr,)


@app.cell
def _(plt, vcorr):
    plt.plot(vcorr, linewidth=2)
    plt.xlabel("Time")
    plt.ylabel("Velocity autocorrelation")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Compute the orientation autocorrelation function in time
    """)
    return


@app.cell
def _(an, plt):
    (ocorr, _tc) = an.orientation_autocorr()
    print(f'Orientation autocorrelation time: {_tc} units')
    plt.plot(ocorr, linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('Orientation autocorrelation')
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Compute the divergence of the flow-field for a given frame
    """)
    return


@app.cell
def _(an):
    an.compute_divergence(frame=0, plot=True, show=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Check incompressibility by computing the mean and s.e.m. of the flow-field divergence
    """)
    return


@app.cell
def _(an):
    divu_means = an.check_imcompressibility()
    print(f"Divergence of velocity: {divu_means.mean()} \u00b1 {divu_means.std()}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Find the centroids and orientations of the $\pm 1/2$ defects in a given frame
    """)
    return


@app.cell
def _(an):
    [cp, cm, phi_p, phi_m] = an.find_defects(frame=5, plot=True, show=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Compute the time-series of the number of $\pm 1/2$ defects
    """)
    return


@app.cell
def _(an):
    [num_p, num_m] = an.num_defects_all()

    print(f"Number of defects per frame:: {(num_p + num_m).mean()} \u00b1 {(num_p + num_m).std()} ")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Optionally delete the downloaded test dataset

    This is behind a button rather than running on its own: marimo orders cells by
    their dataflow, and a bare `remove` call depends only on the path, not on the
    analysis above, so nothing would guarantee it ran last.
    """)
    return


@app.cell
def _(mo):
    delete_button = mo.ui.run_button(label="Delete the downloaded test dataset")
    delete_button
    return (delete_button,)


@app.cell
def _(delete_button, mo, output):
    mo.stop(
        not delete_button.value,
        mo.md(f"Dataset kept at `{output}`."),
    )
    if output.exists():
        output.unlink()
    mo.md(f"Deleted `{output}`.")
    return


if __name__ == "__main__":
    app.run()
