import numpy as np
import dipy.reconst.dti as dti
import dipy.reconst.sfm as sfm
import dipy.core.geometry as geo

def l2norm(vector):
    """
    Normalize a vector to be unit length

    Parameters
    ----------
    vector : 1d array

    Returns
    -------
    The vector, normalized by its norm    
    """
    return vector / np.dot(vector, vector)


def distance_weight(dist, tau=1.5):
    """
    A weighting for the distance from V0 
    """
    return np.exp(-dist/tau)


def weighting(location, out_dir, in_dir, tau=0.5, exp=0.01):
    """
    A weighting that takes into account the distance from V0, as well as the
    angle between the vector between the center of V0 and each of the
    neighboring voxels and the input (column) and output (row) directions of
    the matrix

    """
    norm_location = l2norm(location)
    out_corr = np.dot(norm_location, out_dir) 
    in_corr = np.dot(norm_location, in_dir)
    return (distance_weight(np.dot(location, location)) *
            (np.abs(out_corr) ** exp) * (np.abs(in_corr) ** exp))


def design_matrix(gtab, sphere, evals=np.array([0.0015, 0.0005, 0.0005])):
    """ 

    """
    # Get the generic SFM design matrix:
    sfm_dm = sfm.sfm_design_matrix(gtab, sphere, response=evals)
    # Initialize it with the V0 matrix (which is the original SFM matrix):
    dm = [sfm_dm]
    coords = [0, 1, -1]
    rows = np.arange(np.sum(~gtab.b0s_mask))
    columns = np.arange(sphere.x.shape[0])
    for x in coords:
        for y in coords:
            for z in coords:
                if [x, y, z] == [0, 0, 0]:
                    pass
                else:
                    # Start with the original matrix and downweight each
                    # component as necessary:
                    this_dm = sfm_dm.copy() 
                    location = np.array([x, y, z])
                    for row in rows:
                        out_dir = gtab.bvecs[~gtab.b0s_mask][row]
                        for col in columns:
                            in_dir = sphere.vertices[col]
                            this_dm[row, col] *=\
                                weighting(location, out_dir, in_dir)
                    dm.append(this_dm)
    return dm


def preprocess_signal(data, gtab, i, j, k):
    sig = []
    coords = [0, 1, -1]
    
    for x in coords:
        for y in coords:
            for z in coords:
                location = np.array([x, y, z])
                this_data = data[i+x, j+y, k+z]
                this_data = (this_data[~gtab.b0s_mask]
                             / np.mean(this_data[gtab.b0s_mask]))
                this_data = this_data - np.mean(this_data)
                if np.all(location == np.array([0, 0, 0])):
                    sig.append(this_data)
                else: 
                    sig.append(distance_weight(np.dot(location, location)) *
                              this_data)
    return sig
    
