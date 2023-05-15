"""A module which contains all of the functions used in the SALC notebook"""

import numpy as np

def rotation(axis: np.array, theta: float) -> np.array:
    """
    return a rotation matrix about the given axis by the given angle, theta

    Parameters
    ----------
    axis: np.array
        np.array with shape (3,) which represents the axis of rotation
    theta: float
        rotation angle

    Returns 
    -------
    rot_matrix: np.array
        rotation matrix with shape (3, 3)
    """
    #Ensure that the axis of rotation is normalized
    axis = np.array(axis, dtype=np.float64)
    axis /= np.sqrt(np.dot(axis, axis))
    
    s = np.sin(theta)
    c = np.cos(theta)

    rot_matrix = np.array([[c + (axis[0]**2)*(1-c),             axis[0]*axis[1]*(1-c) - axis[2]*s,  axis[0]*axis[2]*(1-c) + axis[1]*s],
                           [axis[0]*axis[1]*(1-c) + axis[2]*s,  c + (axis[1]**2)*(1-c),             axis[1]*axis[2]*(1-c) - axis[0]*s],
                           [axis[0]*axis[2]*(1-c) - axis[1]*s,  axis[1]*axis[2]*(1-c) + axis[0]*s,  c + (axis[2]**2)*(1-c)           ]])
    return rot_matrix


def reflection(normal: np.array) -> np.array:
    """
    return a reflection matrix across a plane normal to the given vector

    Parameters
    ----------
    normal: np.array
        np.array with shape (3,) which represents the axis of rotation

    Returns 
    -------
    ref_matrix: np.array
        reflection matrix with shape (3, 3)
    """
    #normalize normal vector
    normal = np.array(normal, dtype=np.float64)
    normal /= np.sqrt(np.dot(normal, normal))

    #construct an orthonormal basis which includes the normal vector
    basis = [normal, np.array([1.,0.,0.]), np.array([0.,1.,0.]), np.array([0.,0.,1.])]
    ortho_basis = gram_schmidt(basis)
    P = np.vstack(ortho_basis)
    diag = np.array([[-1., 0., 0.],
                     [ 0., 1., 0.],
                     [ 0., 0., 1.]])
    ref_matrix = P.T@diag@P
    return ref_matrix


def affine(ortho: np.array, trans: np.array) -> np.array:
    """
    create a 3D affine transformation matrix given an orthogonal transformation matrix and a translation vector

    Parameters
    ----------
    ortho: np.array
        orthogonal transformation matrix in the form of an np.array with shape (3,3) 

    trans:  np.array
        translation vector in the form of an np.array with shape (3,)

    Returns 
    -------
    affine_matrix: np.array
        affine transformation matrix with shape (4, 4)
    """
    affine_matrix = np.zeros((4,4), dtype=np.float64)
    affine_matrix[:3, :3] = ortho
    affine_matrix[:-1,3] = np.array(trans, dtype=np.float64)
    affine_matrix[3,3] = 1.
    return affine_matrix
    

def proj(u: np.array, v: np.array, eps=0.0001) -> np.array:
    """projection of the vector v onto the vector u"""

    norm_sq = np.dot(u,u)
    if norm_sq < eps:
        return np.zeros(3, dtype=np.float64)
    else:
        return (np.dot(u,v)/np.dot(u,u))*u


def gram_schmidt(basis: list, eps=0.0001) ->list:
    """return an orthogonal basis set via the gram-schmidt process"""

    #normalize basis
    for i, v in enumerate(basis):
        basis[i] = basis[i].astype(np.float64)
        basis[i] /= np.sqrt(np.dot(v, v))

    #gram-schmidt algorithm
    ortho_basis = []
    for i, v in enumerate(basis):
        ortho_basis.append(v)
        for j in range(i):
            ortho_basis[i] -= proj(basis[j], v, eps=eps)

    #remove zero vectors and normalize orthogonal basis
    zero_indices = []
    for i, v in enumerate(ortho_basis):
        norm_sq = np.dot(v,v)
        if norm_sq < eps:
            zero_indices.append(i)
        else:
            ortho_basis[i] /= np.sqrt(norm_sq)
    for i in zero_indices:
        ortho_basis.pop(i)

    return ortho_basis

def array_equal(a1: np.array, a2: np.array, eps=0.0001) -> np.array:
    """
    Check if two arrays are equal up to a accuracy factor eps

    Parameters
    ----------
    a1 : np.array
        the first array

    a2 : np.array
        the second array

    eps: float
        check that each element of the arrays differ by no more than eps

    Return
    ------
    answer : bool
        returns true if the arrays are equal
    """
    diff = np.abs(a1 - a2)
    return np.all(diff < eps)


def permutation_matrix(vertices, transformed_vertices):
    """generate a permutation matrix associated with a transformation matrix acting on a set of vertices"""
    _, cols = vertices.shape
    perm_mat = np.zeros((cols, cols), dtype=np.int8)

    for i in range(cols):
        for j in range(cols):
            if array_equal(vertices[:,i], transformed_vertices[:,j]):
                perm_mat[i,j] = 1
                break
    
    return perm_mat


def valid_permutation_matrix(perm_mat):
    """check if the given matrix is a valid permutation matrix"""
    rows, cols = perm_mat.shape
    
    for i in range(rows):
        if not (np.sum(perm_mat[i,:]) == 1):
            return False

    for j in range(cols):
        if not (np.sum(perm_mat[:,j]) == 1):
            return False
        
    return True


def conjugate(a: np.array, b: np.array) -> np.array:
    """Return the conjugate of a matrix 'a' by a matrix 'b'"""
    return a @ b @ np.linalg.inv(a)