"""
SCM: The Spatial Continuity Model
---------------------------------

Models diffusion MRI data, assuming that white matter fibers have a spatial
extent, and that we can make assumptions about the data in voxel 0, based on
its neighbors on all sides. 

"""
import numpy as np
import utils as ut
import dipy.reconst.sfm as sfm
import dipy.core.geometry as geo
from dipy.reconst.sfm import SparseFascicleModel, SparseFascicleFit
from dipy.core.onetime import auto_attr
import dipy.core.ndindex as dnd

TAU = 1

class SpatContModel(SparseFascicleModel):
    def __init__(self, gtab, sphere=None, response=[0.0015, 0.0005, 0.0005],
                 solver='ElasticNet', l1_ratio=0.5, alpha=0.001, tau=TAU):
        
        SparseFascicleModel.__init__(self, gtab, sphere, response,
                                     solver, l1_ratio, alpha)
        # We need to retain the original alpha, because we will tweak it in
        # every voxel, depending on the weights applied to the data.
        self.orig_alpha = alpha
        self.tau = tau
    
    @auto_attr
    def sc_design_matrix(self):
        """
        The spatial continuity design matrix, which is used to jointly fit data
        from a voxel and its neighbors at the same time.
        """        
        return ut.design_matrix(self.gtab, self.sphere, self.tau, self.response)

    def fit(self, data, mask=None):
        """
        Fit the SpatContModel object to data

        Parameters
        ----------
        data : array
            The measured signal.

        mask : array, optional
            A boolean array used to mark the coordinates in the data that
            should be analyzed. Has the shape `data.shape[:-1]`. Default: the
            entire volume, except the edges.

        Returns
        -------
        SpatContFit

        """
        if mask is None:
            mask = np.ones(data.shape)
        # Pre-allocate the final result:
        beta = np.zeros(data.shape[:-1] + (self.design_matrix.shape[-1], ))
        # We will go voxel by voxel, avoiding the edges altogether:
        coords = np.array(list(dnd.ndindex((data.shape[0]-2,
                                            data.shape[1]-2,
                                            data.shape[2]-2)))) + 1

        S0 = np.mean(data[..., self.gtab.b0s_mask], -1)
        # Normalize and remove mean:
        norm_data = (data[..., ~self.gtab.b0s_mask] / S0[..., None])
        mean_data = np.mean(norm_data, -1)
        norm_data = norm_data - mean_data[..., None]
        # Weights for the 3x3 matrix of voxels around the voxel of interest,
        # calculated by the distance from the center of the central voxel:
        dist_weights = ut.signal_weights(self.gtab, self.tau)
        dw_shape = np.sum(~self.gtab.b0s_mask)
        for i, j, k in coords:
            local_mask = mask[i-1:i+2, i-1:i+2, i-1:i+2]
            # Don't bother if there's nothing to analyze here:
            if np.any(local_mask):
                local_data = norm_data[i-1:i+2, i-1:i+2, i-1:i+2]
                local_dist_weight = (dist_weights *
                                     local_mask[..., None]).ravel()
                local_dm = (self.design_matrix * local_mask[..., None, None])
                local_dm = local_dm.reshape(-1,self.sc_design_matrix.shape[-1])
                fit_it = local_data.ravel() * local_dist_weight
                this_alpha = (self.orig_alpha *
                          np.sum(local_dist_weight**2)/local_dm.shape[0])
                self.solver.alpha = this_alpha
                beta[i, j, k] = self.solver.fit(local_dm,
                                                fit_it).coef_

        return SpatContFit(self, beta, S0, mean_data)

        
class SpatContFit(SparseFascicleFit):
    def __init__(self, model, beta, S0, mean_signal):
        SparseFascicleFit.__init__(self, model, beta, S0, mean_signal)
