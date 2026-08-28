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
    # Model discovery on an active nematics dataset

    In this example, we perform model discovery on an active nematics dataset generated
    from numerical simulations.

    This dataset contains a trajectory of an active nematic which has developed
    turbulence. This data includes the $Q_{xx}$, $Q_{xy}$, $u_x$ and $u_y$ values at
    all points in space and time.

    Methods are as described in the manuscript: https://arxiv.org/abs/2202.12854

    ### Analysis includes:

    * Using the integral formulation to obtain the optimal model for the Qxx, Qxy and
      vorticity equations in the presence of 5% noise. We will note here that the
      vorticity equations has a bad R-squared score.
    * Using the weak formulation to obtain the optimal flow equation with a much better
      R-squared score.
    * Performing a strong form check of the obtained equation by computing the terms
      and making a spatial comparison.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Importing essential modules
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np

    return np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Importing actnempy modules

    The core object for model identification is called `Anise`, short for Active
    Nematic Identification of Sparse Equations
    """)
    return


@app.cell
def _():
    from actnempy.SINDy import Benchmark

    return (Benchmark,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `Benchmark` is a subclass of `Anise`

    It contains essentially the same methods as `Anise`, with additional benchmarking
    methods like noise addition that will be useful to test on simulation data. Both
    these classes are initialized with the path to the data directory.

    This directory must contain the following files:

    * `processed_data.npz`: A single .npz file containing 4 arrays: `Qxx_all`,
      `Qxy_all`, `u_all` and `v_all`, each of dimensions (NX, NY, NT), with X-Y being
      the spatial dimensions and T being the time dimension. The preprocessing of the
      experimental / simulation data into this format is done elsewhere. (Due to the
      large size, I am not uploading a sample processed data file here, but I will
      happily share it with you if you reach out to me at chaitanya@brandeis.edu!)
    * `metadata.json` : A json file containing three keys: 'dx', 'dy' and 'dt',
      specifying the spatial and temoral discretization of the data.
    * `sindy_library_specs.json`: A json file specifying the parameters for the
      integral formulation, such as maximum function order / derivative order for
      various terms, as well as number of windows and window size. An example json
      file is already provided under `../TestData/`
    """)
    return


@app.cell
def _(Benchmark, mo):
    data_dir = mo.notebook_dir().parent / "TestData"
    an = Benchmark(data_dir)
    return (an,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### We will now visualize the first frame of the dataset
    """)
    return


@app.cell
def _(an):
    an.visualize(10, save=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Let's now add some noise to the dataset as a test for our algorithm
    """)
    return


@app.cell
def _(an):
    an.add_noise_all(0.05)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Let's visualize our noisy fields
    """)
    return


@app.cell
def _(an):
    an.visualize(1, save=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Now, we will apply the integral formulation to this dataset

    The `sindy_int` method performs this operation on `Qxx`, `Qxy` and `ω` in one
    fell swoop.

    As a reference, the equations used for the simulation in this dataset are:

    For the Q-tensor:
    $$
        \partial_t \mathbf{Q} + \nabla\cdot(\mathbf{u}\mathbf{Q}) +
        (\mathbf{\Omega}\cdot\mathbf{Q} - \mathbf{Q}\cdot\mathbf{\Omega})  =
        \lambda \mathbf{E} + \mathbf{H}
    $$

    with

    $$
        \Omega_{ij} = \frac{1}{2}(\partial_i u_j - \partial_j u_i)
    $$
    $$
        E_{ij} = \frac{1}{2}(\partial_i u_j + \partial_j u_i)
    $$
    $$
        H_{ij} = (-a_2-a_4 Q_{kl}Q_{lk}) Q_{ij} + K \partial_k \partial_k Q_{ij}
    $$

    and for the Stokes equation for the flow:
    $$
    \eta\nabla^2 \mathbf{u} = \nabla P + \Gamma \mathbf{u} + \alpha \nabla\cdot \mathbf{Q}
    $$

    The parameters used in the simulation are
    #### $\eta = K = 1$,
    #### $\alpha = 0.3$
    #### $\Gamma = 0.03$
    #### $a_2 = -0.3$
    #### and
    #### $a_4 = 1.36$
    """)
    return


@app.cell
def _(an):
    an.sindy_int()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The SINDy operation produces a `PDE` object

    One for each equation, `pde_Qxx`, `pde_Qxy` and `pde_St`. This object has some
    convenient functionality to analyze the result.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Let's plot the optimality curve for `Qxx` for upto 30 terms

    We use the `plot_fvu` method on the PDE.
    """)
    return


@app.cell
def _(an):
    an.pde_Qxx.plot_fvu(30)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### We clearly see a steep shoulder at n=10

    This is as indicated in the result from `sindy_int`. We can display the model at
    any `n` using the `display_model` method.
    """)
    return


@app.cell
def _(mo):
    num_terms = mo.ui.slider(1,30, step=1, value=10, label="Number of terms")
    num_terms
    return (num_terms,)


@app.cell
def _(an, num_terms):
    an.pde_Qxx.display_model(num_terms.value)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The flow equation is inaccurate despite the noise

    While the `Qxx` and `Qxy` results are good, the $\alpha$ value is off by ~30% and
    the substrate friction term is absent.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### We will now use the weak form for the flow equation with a large window size.
    """)
    return


@app.cell
def _(an):
    an.weak_form(num_windows=50, window_size=(45, 45, 101))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### This gives a much better result than the integral formulation!
    """)
    return


@app.cell
def _(an):
    an.pde_St_w.plot_fvu(5)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `hierarchy` provides the terms in decreasing order of prominence
    """)
    return


@app.cell
def _(an):
    an.pde_St_w.hierarchy()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Testing whether the weak form equation holds in the strong form

    We now compute the correlation between $\nabla\times\nabla\cdot Q$ and
    $\nabla\times \nabla^2 \vec{u}$.
    """)
    return


@app.cell
def _(np):
    from scipy.ndimage import gaussian_filter
    from tqdm import tqdm

    def spatialcorr(a, b):
        ab = a * b
        return np.sum(ab) / np.sqrt(np.sum(a**2) * np.sum(b**2))


    def average(f):
        return gaussian_filter(f, sigma=3)

    return average, spatialcorr, tqdm


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Note above the value of `sigma` used for the averaging filter

    Lowering this value will result in a much larger error in
    $\nabla\times \nabla^2 \vec{u}$ due to the 3 derivatives.
    """)
    return


@app.cell
def _(an, average, np, plt, spatialcorr, tqdm):
    NT = an.NT
    corrs = np.zeros(NT)
    for i in tqdm(range(NT)):
        Qxx = an.Qxx_all[:, :, i]
        Qxy = an.Qxy_all[:, :, i]

        Qxx = average(Qxx)
        Qxy = average(Qxy)

        Q = np.array([[Qxx, Qxy], [Qxy, -Qxx]])
        grid2D = an.grid2D

        divQ = grid2D.div(Q)
        curl_divQ = grid2D.curl(divQ)

        u = an.u_all[:, :, i]
        v = an.v_all[:, :, i]

        u = average(u)
        v = average(v)

        vel = np.array([u, v])
        lap_vel = grid2D.lap(vel)
        curl_lap_vel = grid2D.curl(lap_vel)

        # Number of pixels to remove from the sides due to inaccuracy of derivatives at the boundary
        skip = 5

        left = curl_divQ[skip:-skip, skip:-skip]
        right = curl_lap_vel[skip:-skip, skip:-skip]

        corrs[i] = spatialcorr(left, right)


    figure_mosaic = """
                    AB
                    CC
                    """
    fig, ax = plt.subplot_mosaic(mosaic=figure_mosaic, figsize=(10, 8), facecolor="white")

    plt.sca(ax["A"])
    plt.pcolor(left, cmap="bwr")
    ax["A"].set_aspect("equal")
    plt.title(r"$\nabla\times\nabla\cdot Q$")
    plt.colorbar()
    plt.sca(ax["B"])
    plt.pcolor(right, cmap="bwr")
    ax["B"].set_aspect("equal")
    plt.title(r"$\nabla\times \nabla^2 \vec{u}$")
    plt.colorbar()
    plt.sca(ax["C"])
    plt.plot(corrs)
    plt.plot(NT, corrs[-1], "o", color="red", markersize=10, fillstyle="none")
    plt.xlabel("Time")
    plt.ylabel("Correlation")
    # plt.suptitle("")
    # plt.suptitle(f"{c} " + r"$\mu$" + "M ATP")
    plt.show()
    return


if __name__ == "__main__":
    app.run()
