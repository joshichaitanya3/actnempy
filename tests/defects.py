import numpy as np
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt 

module_dir = Path(__file__).parent.parent.absolute() # The modules are two directories up
sys.path.append(module_dir.as_posix())

from actnem import ActNem
data_dir =  module_dir / "TestData"

actnem = ActNem(data_dir) 

[centroids_p, centroids_m, phi_p, phi_m] = actnem.find_defects(plot=True)

np.savez("defects.npz", cp=centroids_p,
                        cm=centroids_m,
                        phi_p=phi_p,
                        phi_m=phi_m)

[num_p, num_m] = actnem.num_defects_all()

np.savez("num_defects.npz", num_p=num_p,num_m=num_m)
