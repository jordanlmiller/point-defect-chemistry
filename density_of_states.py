"""A module for storing functions related to the density of states plots"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from constants import *



def doscar_plot(base: str, address_list: list, label_list: list, fermi_level_list: list, savename: str):
    """Generate density of states plots for each of the charge states given by the address_list and label_list"""

    n = len(address_list)
    fig, axes = plt.subplots(1, n, figsize=(int(2*n),4), dpi=500, tight_layout=True, sharey=True, squeeze=False)

    for i, address in enumerate(address_list):
        
        data = pd.read_csv(base+address, delim_whitespace=True)
        energy = np.array(data["Energy"])
        DOS = np.array(data["DOS"])
        DOS /= np.max(DOS)

        axes[0][i].set_title(label_list[i])
        axes[0][i].get_xaxis().set_visible(False)
        axes[0][i].plot(DOS, energy, color=(0.2, 0.2, 0.2))
        
        #valence band maximum
        axes[0][i].fill_between([0, 1.2], E_VBM, color=(.7, 0, 0, 0.2))
        #Conduction band maximum
        axes[0][i].fill_between([0, 1.2], E_CBM, 12 , color=(0.7, 0.7, 0, 0.2))
        #Fermi Level
        axes[0][i].hlines(fermi_level_list[i], 0, 1, color="k",linewidth=1, linestyle='dashed')
        axes[0][i].text(1.1, fermi_level_list[i], "$E_{f}$", color='k', va='center', ha='center')

        axes[0][i].set_ylim(0, 12)
        axes[0][i].set_xlim(0, 1.2)

    axes[0][0].set_ylabel("Energy (eV)")
    plt.savefig(savename, bbox_inches="tight")
    plt.show()