import numpy as np
import matplotlib.pyplot as plt


def convert_to_optical_densities(rgb, r0, g0, b0):
    od = rgb.astype(float)
    od[:, :, 0] /= r0
    od[:, :, 1] /= g0
    od[:, :, 2] /= b0

    return -np.log(od)


def color_deconvolution(rgb, r0, g0, b0, verbose=False):
    # See: https://gist.github.com/odebeir/5038467
    stain_od = np.asarray([[0.18, 0.20, 0.08],
                           [0.01, 0.13, 0.0166],
                           [0.10, 0.21, 0.29]])

    n = []
    for r in stain_od:
        n.append(r / np.linalg.norm(r))

    normalized_od = np.asarray(n)

    d = np.linalg.inv(normalized_od)

    od = convert_to_optical_densities(rgb, r0, g0, b0)

    od_max = np.max(od, axis=2)
    if verbose:
        plt.figure()
        plt.imshow(od_max > .1)

    # reshape image on row per pixel
    row_od = np.reshape(od, (-1, 3))
    # do the deconvolution
    row_c = np.dot(row_od, d)

    # restore image shape
    c = np.reshape(row_c, od.shape)

    # remove problematic pixels from the the mask
    od_max[np.isnan(c[:, :, 0])] = 0
    od_max[np.isnan(c[:, :, 1])] = 0
    od_max[np.isnan(c[:, :, 2])] = 0
    od_max[np.isinf(c[:, :, 0])] = 0
    od_max[np.isinf(c[:, :, 1])] = 0
    od_max[np.isinf(c[:, :, 2])] = 0

    return od_max, c

# TODO: Color Normalization
