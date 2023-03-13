# actnempy

Analysis suite for 2D active nematics data, written in Python3.

For an example of a code used in the manuscript [Data-driven discovery of active nematic hydrodynamics](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.129.258001) ([arXiv version here](https://arxiv.org/abs/2202.12854)), see [this Jupyter notebook.](examples/SINDy.ipynb)

# Installation

1. Clone this repository:

```
git clone https://github.com/joshichaitanya3/actnempy.git
```

2. Install via `pip`

```
pip install wheel
cd actnempy
pip install .
```

# Usage

Basic usage is showcased under [examples/basic_example.ipynb](examples/basic_example.ipynb)

Analysis of an entire trajectory is shown under [examples/analyze_trajectory.ipynb](examples/analyze_trajectory.ipynb)

Discovering the underlying PDE model from a trajectory using sparse regression methods as detailed in the manuscript [Data-driven discovery of active nematic hydrodynamics](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.129.258001) ([arXiv version here](https://arxiv.org/abs/2202.12854)) is shown under [examples/SINDy.ipynb](examples/SINDy.ipynb)

_The model identification work detailed in the manuscript was supported by the Department of Energy (DOE) DE-SC0022291. Preliminary data and analysis were supported by the National Science Foundation (NSF) DMR-1855914 and the Brandeis Center for Bioinspired Soft Materials, an NSF MRSEC (DMR-2011846). Computing resources were provided by the NSF XSEDE allocation TG-MCB090163 (Stampede and Comet) and the Brandeis HPCC which is partially supported by the NSF through DMR-MRSEC 2011846 and OAC-1920147._
