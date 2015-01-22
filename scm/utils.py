import numpy as np
import dipy.reconst.dti as dti
import dipy.reconst.sfm as sfm
import dipy.core.geometry as geo
import dipy.core.ndindex as dnd

TAU = 1

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


def distance_weight(dist, tau=1.0):
    """
    A weighting for the distance from V0 
    """
    return np.exp(-dist/tau)


def weighting(location, out_dir, in_dir, tau=TAU, exp=0.01):
    """
    A weighting that takes into account the distance from V0, as well as the
    angle between the vector between the center of V0 and each of the
    neighboring voxels and the input (column) and output (row) directions of
    the matrix

    """
    norm_location = l2norm(location)
    out_corr = np.dot(norm_location, out_dir) 
    in_corr = np.dot(norm_location, in_dir)
    return (distance_weight(np.dot(location, location), tau))# *
            #(np.abs(out_corr) ** exp) * (np.abs(in_corr) ** exp))


def design_matrix(gtab, sphere, evals=np.array([0.0015, 0.0005, 0.0005])):
    """ 

    """
    # Get the generic SFM design matrix:
    sfm_dm = sfm.sfm_design_matrix(gtab, sphere, response=evals)
    # Initialize it with the V0 matrix (which is the original SFM matrix):
    dm = []
    coords = np.array(list(dnd.ndindex((3, 3, 3)))) - 1
    rows = np.arange(np.sum(~gtab.b0s_mask))
    columns = np.arange(sphere.x.shape[0])
    for x, y, z in coords:
        if [x, y, z] == [0, 0, 0]:
            dm.append(sfm_dm)
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
    return np.concatenate(dm)


def signal_weights(gtab, tau=TAU):
    """
    Calculate the weights to apply to signal vectors (depending on the distance
    of the voxels from the center voxel).
    """
    dw_shape = np.sum(~gtab.b0s_mask)
    dist_weights = np.zeros((3, 3, 3, dw_shape))
    coords = np.array(list(dnd.ndindex((3, 3, 3)))) - 1
    for x, y, z in coords:
        location = np.array([x, y, z])
        this = np.ones(dw_shape)
        if np.all(location == np.array([0, 0, 0])):
            pass
        else:
            this *= distance_weight(np.dot(location, location), tau=tau)
        # We want this to match the array conventions in preprocess_signal:
        dist_weights[location[0]+1, location[1]+1, location[2]+1] = this
    return dist_weights.ravel()


def preprocess_signal(data, gtab, i, j, k, dist_weights=None, tau=TAU):
    if dist_weights is None:
        dist_weights = signal_weights(gtab, tau=tau)
    dw_shape = np.sum(~gtab.b0s_mask)
    sig = np.zeros(27 * dw_shape)
    coords = np.array(list(dnd.ndindex((3, 3, 3)))) - 1
    this_data = data[i + coords[:, 0], j + coords[:, 1], k + coords[:, 2]]
    sig = (this_data[..., ~gtab.b0s_mask] /
           np.mean(this_data[..., gtab.b0s_mask], -1)[..., None])
    sig = sig - np.mean(sig, -1)[..., None]
    sig = sig.ravel() * dist_weights
    return sig


