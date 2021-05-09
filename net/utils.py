import numpy as np
from scipy.signal import convolve2d
from sklearn.metrics import mean_squared_error


def get_random_patchs(LDCT_slice, NDCT_slice, patch_size, whole_size=512):
    whole_h = whole_w = whole_size
    h = w = patch_size

    # patch image range
    hd, hu = h // 2, int(whole_h - np.round(h / 2))
    wd, wu = w // 2, int(whole_w - np.round(w / 2))

    # patch image center(coordinate on whole image)
    h_pc, w_pc = np.random.choice(range(hd, hu + 1)), np.random.choice(range(wd, wu + 1))
    LDCT_patch = LDCT_slice[:, h_pc - hd: int(h_pc + np.round(h / 2)), w_pc - wd: int(w_pc + np.round(h / 2))]
    NDCT_patch = NDCT_slice[:, h_pc - hd: int(h_pc + np.round(h / 2)), w_pc - wd: int(w_pc + np.round(h / 2))]

    return LDCT_patch, NDCT_patch


def get_random_patch(LDCT_slice, patch_size, whole_size=512):
    idx = np.random.randint(0, (whole_size-patch_size), 1)
    idy = np.random.randint(0, (whole_size-patch_size), 1)

    LDCT_patch = LDCT_slice[idx:(idx+patch_size), idy:(idy+patch_size)]

    return idx,idy,LDCT_patch


# 以左上角为index
def get_index_patch(LDCT_slice, X, Y, patch_size):
    LDCT_patch = LDCT_slice[X:X+patch_size, Y:Y+patch_size]

    return LDCT_patch


def cal_psnr(img1, img2):
    mse = (np.abs(img1 - img2)*np.abs(img1 - img2)).mean()
    psnr = 20 * np.log10(65535 / mse)
    return psnr


def cal_ssim(im1, im2):
    assert len(im1.shape) == 2 and len(im2.shape) == 2
    assert im1.shape == im2.shape
    mu1 = im1.mean()
    mu2 = im2.mean()
    sigma1 = np.sqrt(((im1 - mu1) ** 2).mean())
    sigma2 = np.sqrt(((im2 - mu2) ** 2).mean())
    sigma12 = ((im1 - mu1) * (im2 - mu2)).mean()
    k1, k2, L = 0.01, 0.03, 100
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    C3 = C2 / 2
    l12 = (2 * mu1 * mu2 + C1) / (mu1 ** 2 + mu2 ** 2 + C1)
    c12 = (2 * sigma1 * sigma2 + C2) / (sigma1 ** 2 + sigma2 ** 2 + C2)
    s12 = (sigma12 + C3) / (sigma1 * sigma2 + C3)
    ssim = l12 * c12 * s12
    return ssim


def PSNR(img1, img2):
    mse = np.square(np.abs(img1 - img2)).mean()
    psnr = 20 * np.log10(1.0 / mse)
    return psnr


def SSIM(img1, img2):
    mean1 = img1.mean()  # 均值
    mean2 = img2.mean()

    sigma1 = np.sqrt(np.square(img1 - mean1).mean())  # 标准差
    sigma2 = np.sqrt(np.square(img2 - mean2).mean())
    sigma12 = ((img1 - mean1)*(img2 - mean2)).mean()  # 协方差

    k1, k2, L = 0.01, 0.03, 1
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    # C3 = C2 / 2

    # lumi = (2*mean1*mean2 + C1) / (np.square(mean1) + np.square(mean2) + C1)
    # cont = (2*sigma1*sigma2 + C2) / (np.square(sigma1) + np.square(sigma2) + C2)
    # stru = (2*sigma12 + C3) / (sigma1*sigma2 + C3)

    # ssim = lumi*cont*stru

    ssim = (2*mean1*mean2 + C1)*(2*sigma12 + C2) / ((np.square(mean1) + np.square(mean2) + C1)*(np.square(sigma1) + np.square(sigma2) + C2))

    return ssim


def RMSE(img1, img2):
    return np.sqrt(np.square(np.abs(img1 - img2)).mean())


def set_window(image, window_idx, window_size):
    image = image*1.0-1024  # 转为HU
    w_min = window_idx - window_size / 2
    w_max = window_idx + window_size / 2

    rows, cols = image.shape[0], image.shape[1]

    for i in range(rows):
        for j in range(cols):
            if image[i][j] < w_min:
                image[i][j] = 0
            elif image[i][j] > w_max:
                image[i][j] = 1
            else:
                image[i][j] = (image[i][j]-w_min)/window_size

    return image


def to01(image):
    rows, cols = image.shape[0], image.shape[1]
    image = image*1.0-1024  # 转为HU
    for i in range(rows):
        for j in range(cols):
            if image[i][j] <= -500:
                image[i][j] = 0
            elif image[i][j] >= 1000:
                image[i][j] = 1.0
            else:
                image[i][j] = image[i][j] / 1500.0
    return image
