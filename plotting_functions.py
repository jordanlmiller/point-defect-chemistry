"""A module to contain the functions which generate the various plots used in the report"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def doscar_plot(base, address_list, savename, label_list, fermi_level_list):
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
    
    
    
def strain_tensor_bar(eigenvalues, charge_states, pg_list, savename):
    
    bar_spacing = 0.85
    group_spacing = 3.5
    
    n = len(eigenvalues) // 3
    xpos = []
    for i in range(n):
        for j in range(3):
            xpos.append(group_spacing*i+(j*bar_spacing))
    
            
    fig, ax = plt.subplots(figsize=(int(2*n),4), dpi=300)
    fig.tight_layout()
    max_value = np.max(eigenvalues)
    min_value = np.min(eigenvalues)
    if min_value > 0 :
        min_value = 0
    
    text_offset = (max_value - min_value)*0.1
    
    for i in range(n):
        point_group = str(pg_list[i])
        first_letter = point_group[0]
        subscript = point_group[1:]
        pg_string = "${"+first_letter+"}_{"+subscript+"}$"
        eigvals = eigenvalues[3*i:3*i+3]
        extremal_sign = np.sign(eigvals)[np.argmax(np.abs(eigvals))]
        text_height = np.max(np.abs(eigvals))*extremal_sign+text_offset*extremal_sign
        plt.text(xpos[3*i+1], text_height, pg_string, ha='center')
        
    # Save the chart so we can loop through the bars below.
    bars = ax.bar(x=xpos, height=eigenvalues, color="#FFAA33")      
    
    plt.ylim(min_value-(2*text_offset), max_value+(2*text_offset))

    # Axis formatting.
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')
    ax.tick_params(bottom=False, left=False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color='#EEEEEE')
    ax.xaxis.grid(False)
    ax.set_title("Strain Tensor Eigenvalues")
    ax.plot([-bar_spacing, max(xpos)+bar_spacing], [0,0], color='k')
    ax.set_xticks(xpos[1::3])
    ax.set_xticklabels(charge_states)
    ax.set_xlabel("Charge State")
    ax.set_xlim(-bar_spacing, max(xpos)+bar_spacing)
    ax.set_ylim(0,)
    plt.savefig(savename+"_bar.pdf", bbox_inches='tight')
    plt.show()  
    plt.close()
    

def displacement_plot(atomic_pos, ref_pos, num_Mg, num_O, arrow_length=4):
    O_color = (.7, 0, 0, 0.8)
    Mg_color = (0.7, 0.7, 0, 0.8)
    color = [Mg_color]*num_Mg + [O_color]*num_O
    dx, dy, dz = displacement_field(atomic_pos, ref_pos)
    ax = plt.figure(figsize=(3,3), dpi=300).add_subplot(projection='3d')
    ax.quiver(ref_pos[0], ref_pos[1], ref_pos[2], dx, dy, dz, normalize=False, length=arrow_length, color='k')
    ax.scatter(ref_pos[0], ref_pos[1], ref_pos[2], c=color)
    plt.show()
    plt.close()
