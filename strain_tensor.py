"""A module which contains functions that calculate the strain tensor associated with a defect site using a finite difference approximation"""

import numpy as np
import matplotlib.pyplot as plt
from pymatgen.symmetry.analyzer import PointGroupAnalyzer
from pymatgen.core.structure import Molecule

from constants import *

def minimum_image(x):
    """return the minimum image of a vector under periodic boundary conditions"""
    x[x < - (L * 0.5)] = x[x < - (L * 0.5)] + L
    x[x >=  (L * 0.5)] = x[x >=  (L * 0.5)] - L
    return x



def difference_from_point(p, vector_field):
    """return the periodic difference from a vector field to a point"""
    dx = minimum_image(p[0] - vector_field[0])
    dy = minimum_image(p[1] - vector_field[1])
    dz = minimum_image(p[2] - vector_field[2])
    return dx, dy, dz



def difference_vector(u, v):
    """#return the periodic difference of two vectors"""
    return minimum_image(u - v)



def displacement_field(atomic_pos, ref_pos):
    """#return tuple of displacement vectors associated with each atomic position"""
    dx = difference_vector(atomic_pos[0], ref_pos[0])
    dy = difference_vector(atomic_pos[1], ref_pos[1])
    dz = difference_vector(atomic_pos[2], ref_pos[2])
    return dx, dy, dz



def periodic_distance(x, ref_pos):
    """calculate the periodic distance between points in space"""
    dx, dy, dz = difference_from_point(x, ref_pos)
    return np.sqrt((dx**2) + (dy**2) + (dz**2))



def nearest_neighbor_indices(defect_pos, ref_pos):
    """find the indices associated with the nearest neighbors of the defect site"""
    distance = periodic_distance(defect_pos, ref_pos)
    sorted_indices = np.argsort(distance)
    num_zeros = np.sum(distance  == 0.)
    num_nearest_neighbors = np.sum(distance == distance[sorted_indices][num_zeros:][0])
    return sorted_indices[num_zeros:num_nearest_neighbors+num_zeros]



def get_atomic_index(atom_pos, ref_pos):
    """get the index associated with a particular position in the reference"""
    distance = periodic_distance(atom_pos, ref_pos)
    return np.argwhere(distance == 0.)[0][0]       



def generate_displacement_gradient(defect_pos, atomic_pos, ref_pos):
    """generate the displacement gradient at the defect site with reference to the unrelaxed crystal"""
    
    nn_indices = nearest_neighbor_indices(defect_pos, ref_pos)
    disp_field = displacement_field(atomic_pos, ref_pos)
    
    #generate jacobian of displacement field at defect site
    jac_list = []
    for dim in range(3):
        row_vec = np.zeros(3)
        for index in nn_indices:
            
            q = np.array([ref_pos[0][index], ref_pos[1][index], ref_pos[2][index]]) - defect_pos
            q = minimum_image(q)
            opp_index = get_atomic_index(minimum_image(defect_pos - q), ref_pos)
            h = np.sqrt(np.dot(q, q))
            q /= h
            dudq =  0.5 * (disp_field[dim][index] - disp_field[dim][opp_index]) / h
            row_vec += dudq * q
        
        jac_list.append(row_vec / len(nn_indices))

    jacobian = np.vstack(jac_list)
    return jacobian 



def point_group(atomic_pos: np.ndarray, defect_pos: np.ndarray):
    """Return the point group of the coordination environment surronding a defect site"""
    pos = difference_from_point(defect_pos, atomic_pos)
    pos_list = []
    n = len(pos[0])
    for i in range(n):
        pos_list.append(np.array([pos[0][i], pos[1][i], pos[2][i]]))
    m = Molecule(["Mg"]*n, pos_list)
    pg =  PointGroupAnalyzer(m).get_pointgroup()
    return pg



def generate_strain_tensor(defect_pos, atomic_pos, ref_pos):  
    """#generate the symmetric strain tensor at the defect site with reference to the unrelaxed crystal"""     

    jacobian = generate_displacement_gradient(defect_pos, atomic_pos, ref_pos)

    #generate strain tensor from jacobian
    strain_tensor = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            strain_tensor[i, j] = 0.5 * (jacobian[i,j] + jacobian[j, i])
    return strain_tensor



def distance_matrix(pos: np.ndarray, defect_pos: np.ndarray) -> np.ndarray:
    """generate a distance matrix"""
    pos = difference_from_point(defect_pos, pos)
    n = pos[0].shape[0]
    dist_mat = np.zeros((n,n), dtype=np.float64)
    for i in range(n):
        for j in range(i+1, n):
            d = np.array([pos[0][i], pos[1][i], pos[2][i]]) - np.array([pos[0][j], pos[1][j], pos[2][j]])
            dist = np.sqrt(np.dot(d, d))
            dist_mat[i, j] = dist
            dist_mat[j, i] = dist
    return dist_mat


      
def nearest_neighbor_distance(defect_pos: np.ndarray, ref_pos: np.ndarray, atomic_pos: np.ndarray) -> float:
    """calculate the average nearest neighbor distance"""
    nn_indices = nearest_neighbor_indices(defect_pos, ref_pos)
    nn_pos = atomic_pos[0][nn_indices], atomic_pos[1][nn_indices], atomic_pos[2][nn_indices]
    test = ref_pos[0][nn_indices], ref_pos[1][nn_indices], ref_pos[2][nn_indices]
    distance = np.around(distance_matrix(nn_pos, defect_pos), 3).flatten()
    sorted_indices = np.argsort(distance)
    num_zeros = np.sum(distance  == 0.)
    num_edges = 24
    avg_dist = np.mean(distance[sorted_indices[num_zeros:num_edges+num_zeros]])
    return avg_dist



def strain_tensor_bar(pos_list: list, label_list: list, defect_pos: np.ndarray, ref_pos: tuple, savename: str):
    """Generate a bar chart of the strain tensor values at the defect site for each charge state
    Parameters
    ----------
    pos_list: list
        list of atomic positions in the supercell as numpy arrays

    label_list: list
        list of labels for each subplot of the bar chart

    defect_pos: np.ndarray
        numpy array with shape (3,) containing the position of the defect of prior to relaxation

    ref_pos: tuple
        tuple of the x, y, z positions of the atoms in the supercell prior to relaxation. Each element of the tuple is a numpy array with shape (n,) where n is the number of atoms in the supercell

    savename: str
        name used for saving the plot

    """
    strain_eigenvalues = []
    pg_list = []
    for i, pos in enumerate(pos_list):
        
        atomic_pos = (lattice_constant*np.array(pos["X"]), 
                      lattice_constant*np.array(pos["Y"]), 
                      lattice_constant*np.array(pos["Z"]))

        #strain calculations
        strain_tensor = generate_strain_tensor(defect_pos, atomic_pos, ref_pos)
        strain_eigenvalues += list(np.linalg.eigvals(strain_tensor))        
        nn_indices = nearest_neighbor_indices(defect_pos, ref_pos)
        atomic_pos = atomic_pos[0][nn_indices], atomic_pos[1][nn_indices], atomic_pos[2][nn_indices]
        pg = str(point_group(atomic_pos, defect_pos))
        first_letter = pg[0]
        subscript = pg[1:]
        pg_string = "${"+first_letter+"}_{"+subscript+"}$"
        pg_list.append(pg_string)

    #Bar Chart  
    bar_spacing = 0.85
    group_spacing = 3.5
    
    n = len(pos_list)
    xpos = []
    for i in range(n):
        for j in range(3):
            xpos.append(group_spacing*i+(j*bar_spacing))
      
    fig, ax = plt.subplots(figsize=(int(2*n),4), dpi=300)
    fig.tight_layout()
    max_value = np.max(strain_eigenvalues)
    min_value = np.min(strain_eigenvalues)
    if min_value > 0 :
        min_value = 0
    
    text_offset = (max_value - min_value)*0.1
    
    for i in range(n):

        eigvals = strain_eigenvalues[3*i:3*i+3]
        extremal_sign = np.sign(eigvals)[np.argmax(np.abs(eigvals))]
        text_height = np.max(np.abs(eigvals))*extremal_sign+text_offset*extremal_sign
        plt.text(xpos[3*i+1], text_height, pg_list[i], ha='center')
        
    # Save the chart so we can loop through the bars below.
    bars = ax.bar(x=xpos, height=strain_eigenvalues, color="#FFAA33")       
    plt.ylim(min_value-(2*text_offset), max_value+(2*text_offset))

    # Plot Formatting.
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
    ax.set_xticklabels(label_list)
    ax.set_xlabel("Charge State")
    ax.set_xlim(-bar_spacing, max(xpos)+bar_spacing)
    ax.set_ylim(0,)
    plt.savefig(savename+"_bar.pdf", bbox_inches='tight')
    plt.show()  
    plt.close()