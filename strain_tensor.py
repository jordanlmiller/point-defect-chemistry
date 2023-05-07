"""A module which contains functions that calculat the strain tensor associated with a defect site usisng a finite difference approximation"""

import numpy as np
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



def generate_strain_tensor(defect_pos, atomic_pos, ref_pos):  
    """#generate the symmetric strain tensor at the defect site with reference to the unrelaxed crystal"""     

    jacobian = generate_displacement_gradient(defect_pos, atomic_pos, ref_pos)

    #generate strain tensor from jacobian
    strain_tensor = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            strain_tensor[i, j] = 0.5 * (jacobian[i,j] + jacobian[j, i])
    return strain_tensor