import numpy as np
from PIL import Image
from scipy.ndimage import median_filter, grey_opening, grey_closing


def to_grayscale(img_arr):
    if img_arr.ndim == 3:
        return np.dot(img_arr[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.float32)
    return img_arr.astype(np.float32)


def adaptive_median_filter(img, max_window=7):
    # Approximate adaptive median filter
    # Start with 3x3, then 5x5, then 7x7
    out = img.copy()
    for w in [3, 5, max_window]:
        med = median_filter(out, size=w)
        out = med
    return out


def anisotropic_diffusion(img, niter=10, kappa=30, gamma=0.1):
    # Perona-Malik diffusion
    img = img.astype(np.float32)
    for _ in range(niter):
        nablaN = np.roll(img, -1, axis=0) - img
        nablaS = np.roll(img, 1, axis=0) - img
        nablaE = np.roll(img, -1, axis=1) - img
        nablaW = np.roll(img, 1, axis=1) - img

        cN = np.exp(-(nablaN / kappa) ** 2)
        cS = np.exp(-(nablaS / kappa) ** 2)
        cE = np.exp(-(nablaE / kappa) ** 2)
        cW = np.exp(-(nablaW / kappa) ** 2)

        img = img + gamma * (cN * nablaN + cS * nablaS + cE * nablaE + cW * nablaW)
    return img


def hist_equalize(img):
    # Standard histogram equalization for grayscale
    img = img.astype(np.uint8)
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    return cdf[img]


def otsu_threshold(img):
    # Compute Otsu threshold
    hist, _ = np.histogram(img.flatten(), bins=256, range=(0, 256))
    total = img.size
    sum_total = np.dot(np.arange(256), hist)
    sumB, wB, wF, var_max, threshold = 0.0, 0.0, 0.0, 0.0, 0

    for t in range(256):
        wB += hist[t]
        if wB == 0:
            continue
        wF = total - wB
        if wF == 0:
            break
        sumB += t * hist[t]
        mB = sumB / wB
        mF = (sum_total - sumB) / wF
        var_between = wB * wF * (mB - mF) ** 2
        if var_between > var_max:
            var_max = var_between
            threshold = t
    return threshold


def apply_morphology(img, kernel=3):
    # grey opening + closing
    opened = grey_opening(img, size=(kernel, kernel))
    closed = grey_closing(opened, size=(kernel, kernel))
    return closed


def preprocess_image(pil_img, cfg):
    img = np.array(pil_img)

    if cfg.get("USE_GRAYSCALE", False):
        img = to_grayscale(img)
    else:
        # use luminance for preprocessing but keep RGB later
        img = to_grayscale(img)

    if cfg.get("APPLY_ADAPTIVE_MEDIAN", False):
        img = adaptive_median_filter(img, max_window=cfg.get("ADAPTIVE_MEDIAN_MAX", 7))

    if cfg.get("APPLY_ANISOTROPIC_DIFFUSION", False):
        img = anisotropic_diffusion(
            img,
            niter=cfg.get("DIFFUSION_ITER", 10),
            kappa=cfg.get("DIFFUSION_KAPPA", 30),
            gamma=cfg.get("DIFFUSION_GAMMA", 0.1),
        )

    if cfg.get("APPLY_HIST_EQ", False):
        img = hist_equalize(img)

    if cfg.get("APPLY_MORPH_OPEN_CLOSE", False):
        img = apply_morphology(img, kernel=cfg.get("MORPH_KERNEL", 3))

    # build mask if requested
    mask = None
    if cfg.get("APPLY_OTSU_MASK", False):
        thr = otsu_threshold(img.astype(np.uint8))
        mask = (img >= thr).astype(np.uint8)

    # convert to 3-channel RGB-like image
    img = np.clip(img, 0, 255).astype(np.uint8)
    if cfg.get("USE_GRAYSCALE", False):
        img = np.stack([img, img, img], axis=-1)
    else:
        img = np.stack([img, img, img], axis=-1)

    return Image.fromarray(img), mask
