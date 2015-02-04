import numpy as np
import dipy.reconst.dti as dti
import dipy.reconst.sfm as sfm
import dipy.core.geometry as geo
import dipy.core.ndindex as dnd

NEIGHBORS = 3


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
    return vector / np.sqrt(np.dot(vector, vector).astype(float))


def distance_weight(dist, tau):
    """
    A weighting for the distance from V0 
    """
    return np.exp(-dist/tau)


def dm_weighting(location, out_dir, in_dir, tau, dir_weighting=0.5):
    """
    A weighting that takes into account the distance from V0, as well as the
    angle between the vector between the center of V0 and each of the
    neighboring voxels and the input (column) and output (row) directions of
    the matrix

    """
    norm_location = l2norm(location)
    out_corr = np.abs(np.dot(norm_location, out_dir))
    in_corr = np.abs(np.dot(norm_location, in_dir))
    dir_factor = (out_corr * in_corr) ** dir_weighting
    return (dir_factor * 
            distance_weight(np.dot(location, location), tau))


def design_matrix(gtab, sphere, tau,
                  evals=np.array([0.0015, 0.0005, 0.0005])):
    """
    The volumized design matrix by voxel in the NEIGHBORS-by-NEIGHBORS
    neighborhood 
    

    """
    # Get the generic SFM design matrix:
    sfm_dm = sfm.sfm_design_matrix(gtab, sphere, response=evals)
    # Initialize it with the V0 matrix (which is the original SFM matrix):
    dm = np.zeros((NEIGHBORS, NEIGHBORS, NEIGHBORS,
                   np.sum(~gtab.b0s_mask), sphere.x.shape[0]))
    coords = np.array(list(dnd.ndindex((NEIGHBORS, NEIGHBORS, NEIGHBORS)))) - 1
    rows = np.arange(np.sum(~gtab.b0s_mask))
    columns = np.arange(sphere.x.shape[0])
    dm[1, 1, 1] = sfm_dm
    for x, y, z in coords:
        location = np.array([x, y, z])
        if np.all(location == np.array([0, 0, 0])):
            pass
        else:
            # Start with the original matrix and downweight each
            # component as necessary:
            for row in rows:
                out_dir = gtab.bvecs[~gtab.b0s_mask][row]
                for col in columns:
                    in_dir = sphere.vertices[col]
                    dm[location[0]+1, location[1]+1, location[2]+1, row, col] =\
                dm_weighting(location, out_dir, in_dir, tau) * sfm_dm[row, col]
    return dm


def signal_weights(gtab, tau):
    """
    Calculate the weights to apply to signal vectors (depending on the distance
    of the voxels from the center voxel).
    """
    dw_shape = np.sum(~gtab.b0s_mask)
    dist_weights = np.zeros((NEIGHBORS, NEIGHBORS, NEIGHBORS, dw_shape))
    coords = np.array(list(dnd.ndindex((NEIGHBORS, NEIGHBORS, NEIGHBORS)))) - 1
    for x, y, z in coords:
        location = np.array([x, y, z])
        this = np.ones(dw_shape)
        if np.all(location == np.array([0, 0, 0])):
            pass
        else:
            this *= distance_weight(np.dot(location, location), tau=tau)
        # We want this to match the array conventions in preprocess_signal:
        dist_weights[location[0]+1, location[1]+1, location[2]+1] = this
    return dist_weights


# XXX The following function should be removed at some point. For now, it's
# useful to keep around:

def preprocess_signal(data, gtab, i, j, k, dist_weights=None, tau=None):
    if dist_weights is None:
        dist_weights = signal_weights(gtab, tau=tau)
    dw_shape = np.sum(~gtab.b0s_mask)
    sig = np.zeros((NEIGHBORS ** 3) * dw_shape)
    coords = np.array(list(dnd.ndindex((NEIGHBORS, NEIGHBORS, NEIGHBORS)))) - 1
    this_data = data[i + coords[:, 0], j + coords[:, 1], k + coords[:, 2]]
    sig = (this_data[..., ~gtab.b0s_mask] /
           np.mean(this_data[..., gtab.b0s_mask], -1)[..., None])
    sig = sig - np.mean(sig, -1)[..., None]
    sig = sig.ravel() * dist_weights.ravel()
    return sig


