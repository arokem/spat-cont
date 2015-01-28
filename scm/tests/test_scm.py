import numpy as np
import numpy.testing as npt
import nibabel as nib

import scm.scm as scm
import dipy.data as dpd
import dipy.core.gradients as grad
import dipy.core.optimize as opt

def test_scm():
    fdata, fbvals, fbvecs = dpd.get_data()
    data = nib.load(fdata).get_data()
    gtab = grad.gradient_table(fbvals, fbvecs)
    scmodel = scm.SpatContModel(gtab)
    mask = np.ones(data.shape[:-1])
    # Fit it with a mask:
    scfit = scmodel.fit(data, mask)
    # And without
    scfit = scmodel.fit(data)
