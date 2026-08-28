# actnempy

[![Test](https://github.com/joshichaitanya3/actnempy/actions/workflows/test.yml/badge.svg)](https://github.com/joshichaitanya3/actnempy/actions/workflows/test.yml)
[![Lint](https://github.com/joshichaitanya3/actnempy/actions/workflows/lint.yml/badge.svg)](https://github.com/joshichaitanya3/actnempy/actions/workflows/lint.yml)

Analysis suite for 2D active nematics data, written in Python3.

This code has benefited from crucial contributions from Matthew S. E. Peterson ([@mattsep](https://github.com/mattsep)) and Michael M. Norton ([@wearefor](https://github.com/wearefor)).

For an example of a code used in the manuscript [Data-driven discovery of active nematic hydrodynamics](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.129.258001) ([arXiv version here](https://arxiv.org/abs/2202.12854)), see [this Jupyter notebook.](examples/SINDy.ipynb)

# Installation

This project uses [uv](https://docs.astral.sh/uv/) to manage dependencies and packaging.

1. Clone this repository:

```
git clone https://github.com/joshichaitanya3/actnempy.git
cd actnempy
```

2. Install with `uv`

```
uv sync
```

This creates a `.venv` with `actnempy` and its dependencies installed. Prefix commands with `uv run` (e.g. `uv run python`) to run them inside that environment, or activate it directly with `source .venv/bin/activate`.

Alternatively, install with `pip` (without `uv`):

```
pip install .
```

# Running Tests

The test suite uses Python's built-in `unittest` framework and can be run with `pytest`.

Install the `dev` extra, which includes `pytest`:

```
uv sync --extra dev
```

Then run all tests from the repository root:

```
uv run pytest
```

To run a specific test file:

```
uv run pytest tests/test_actnem.py -v
```

> [!NOTE]
> `test_actnem.py` downloads a small test dataset (~few MB) from Google Drive on first run via `gdown`. Ensure you have an internet connection, or pre-place the data in `TestData/processed_data.npz`.

# Usage

Basic usage is showcased under [examples/basic_example.ipynb](examples/basic_example.ipynb)

Analysis of an entire trajectory is shown under [examples/analyze_trajectory.ipynb](examples/analyze_trajectory.ipynb)

Discovering the underlying PDE model from a trajectory using sparse regression methods as detailed in the manuscript [Data-driven discovery of active nematic hydrodynamics](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.129.258001) ([arXiv version here](https://arxiv.org/abs/2202.12854)) is shown under [examples/SINDy.ipynb](examples/SINDy.ipynb)

### New: Marimo notebooks

Reactive [Marimo](https://docs.marimo.io/) notebook versions of the Jupyter notebooks described above are now available. They provide additional functionality such as sliders to interactively vary defect detection hyper-parameters in [basic_example.py](examples/basic_example.py) and display model with different number of terms in [SINDy.py](examples/SINDy.py).

These are tutorials, so the demonstration of the library usage is their goal, so open them with `marimo edit`:

```bash
uv run marimo edit examples/
```

Or to open a specific notebook:

```bash
uv run marimo edit examples/basic_example.py  
```

[basic_example.py](examples/basic_example.py) additionally stands on its own as a code-free, read-only app, in which the plot of the director with its annotated defects responds to the defect-detection hyper-parameters:

```bash
uv run marimo run examples/basic_example.py
```

_The model identification work detailed in the manuscript was supported by the Department of Energy (DOE) DE-SC0022291. Preliminary data and analysis were supported by the National Science Foundation (NSF) DMR-1855914 and the Brandeis Center for Bioinspired Soft Materials, an NSF MRSEC (DMR-2011846). Computing resources were provided by the NSF XSEDE allocation TG-MCB090163 (Stampede and Comet) and the Brandeis HPCC which is partially supported by the NSF through DMR-MRSEC 2011846 and OAC-1920147._
