"""A module to contain the functions which generate the various plots used in the report"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from strain_tensor import displacement_field
from constants import *



def displacement_plot(atomic_pos: np.ndarray, ref_pos: np.ndarray, num_Mg: int, num_O: int, arrow_length=4):
    O_color = (.7, 0, 0, 0.8)
    Mg_color = (0.7, 0.7, 0, 0.8)
    color = [Mg_color]*num_Mg + [O_color]*num_O
    dx, dy, dz = displacement_field(atomic_pos, ref_pos)
    ax = plt.figure(figsize=(3,3), dpi=300).add_subplot(projection='3d')
    ax.quiver(ref_pos[0], ref_pos[1], ref_pos[2], dx, dy, dz, normalize=False, length=arrow_length, color='k')
    ax.scatter(ref_pos[0], ref_pos[1], ref_pos[2], c=color)
    plt.show()
    plt.close()
