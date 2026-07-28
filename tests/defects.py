from pathlib import Path

import numpy as np

from actnempy import ActNem

module_dir = Path(__file__).parent.parent.absolute()  # The modules are two directories up
data_dir = module_dir / "TestData"

actnem = ActNem(data_dir)

[centroids_p, centroids_m, phi_p, phi_m] = actnem.find_defects(plot=True)

np.savez("defects.npz", cp=centroids_p, cm=centroids_m, phi_p=phi_p, phi_m=phi_m)

[num_p, num_m] = actnem.num_defects_all()

np.savez("num_defects.npz", num_p=num_p, num_m=num_m)
