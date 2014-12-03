import numpy as np
import numpy.testing as npt

import analysis as ana


def test_out_signal():
    sig1 = ana.kernel_signal(np.array([1, 0, 0]),
                             np.array([1, 0, 0]),
                             np.array([1.5, 0.5, 0.5]), 1)
    sig2 = ana.canonical_tensor(np.eye(3),
                                np.array([1.5, 0.5, 0.5]),
                                np.array([1, 0, 0])) 
    npt.assert_equal(sig1, sig2)
              


def test_design_signal():
    sig1 = ana.design_signal(np.array([0, 0, 0]),
                             np.array([1, 0, 0]),
                             np.array([1, 0, 0]))

    sig2 = ana.kernel_signal(np.array([1, 0, 0]),
                             np.array([1, 0, 0]),
                             np.array([1.5, 0.5, 0.5]), 1)

    npt.assert_equal(sig1, sig2)
