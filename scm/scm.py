"""
SCM: The Spatial Continuity Model

Model diffusion MRI data, assuming that white matter fibers have a spatial
extent, and that we can make assumptions about the data in voxel 0, based on
its neighbors on all sides.

Something like that.


"""
import utils as ut
import dipy.reconst.sfm as sfm
import dipy.core.geometry as geo
from dipy.reconst.sfm import SparseFascicleModel, SparseFascicleFit
from dipy.core.onetime import auto_attr

class SpatContModel(SparseFascicleModel):
    def __init__(self, gtab, sphere=None, response=[0.0015, 0.0005, 0.0005],
                 solver='ElasticNet', l1_ratio=0.5, alpha=0.001):
        
        SparseFascicleModel.__init__(self, gtab, sphere, response,
                                     solver, l1_ratio, alpha)

    @auto_attr
    def design_matrix(self):
        return ut.design_matrix(self.gtab, self.sphere, self.response)
        
    def fit(self, data, mask=None):
        """
        Fit the SpatContModel object to data

        Parameters
        ----------
        data : array
            The measured signal from one voxel.

        mask : array
            A boolean array used to mark the coordinates in the data that
            should be analyzed that has the shape data.shape[-1]

        Returns
        -------
        SpatContFit

        """
        if mask is None:
            flat_data = np.reshape(data, (-1, data.shape[-1]))
        else:
            mask = np.array(mask, dtype=bool, copy=False)
            flat_data = np.reshape(data[mask], (-1, data.shape[-1]))

        # Fitting is done on the relative signal (S/S0):
        flat_S0 = np.mean(flat_data[..., self.gtab.b0s_mask], -1)
        flat_S = flat_data[..., ~self.gtab.b0s_mask] / flat_S0[..., None]
        flat_mean = np.mean(flat_S, -1)
        flat_params = np.zeros((flat_data.shape[0],
                                self.design_matrix.shape[-1]))

        for vox, vox_data in enumerate(flat_S):
            if np.any(np.isnan(vox_data)):
                flat_params[vox] = (np.zeros(self.design_matrix.shape[-1]))
            else:
                fit_it = vox_data - flat_mean[vox]
                flat_params[vox] = self.solver.fit(self.design_matrix,
                                                   fit_it).coef_
        if mask is None:
            out_shape = data.shape[:-1] + (-1, )
            beta = flat_params.reshape(out_shape)
            mean_out = flat_mean.reshape(out_shape)
            S0 = flat_S0.reshape(out_shape).squeeze()
        else:
            beta = np.zeros(data.shape[:-1] +
                            (self.design_matrix.shape[-1],))
            beta[mask, :] = flat_params
            mean_out = np.zeros(data.shape[:-1])
            mean_out[mask, ...] = flat_mean.squeeze()
            S0 = np.zeros(data.shape[:-1])
            S0[mask] = flat_S0

        return SpatContFit(self, beta, S0, mean_out.squeeze())

        
class SpatContFit(SparseFascicleFit):
    def __init__(self, model, beta, S0, mean_signal):
        SparseFascicleFit.__init__(self, model, beta, S0, mean_signal)
