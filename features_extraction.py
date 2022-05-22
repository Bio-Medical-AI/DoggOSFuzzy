from copy import deepcopy

import numpy as np
import cv2
import albumentations as A
from skimage.feature import graycomatrix, graycoprops
from skimage.color import hsv2rgb
from Albument import Albument


def textural_features(image_rgb, mask):
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    x_min = -1
    x_max = -1
    y_min = -1
    y_max = -1
    for i in range(mask.shape[0]):
        if np.any(mask[i, :] > 0):
            if y_min == -1:
                y_min = i
            y_max = i
    for j in range(mask.shape[1]):
        if np.any(mask[:, j] > 0):
            if x_min == -1:
                x_min = j
            x_max = j
    albumentations_augs = []
    albumentations_augs.append(A.Crop(x_min, y_min, x_max, y_max, p=1.0))
    albumentations_augs = A.Compose(albumentations_augs)
    albument = Albument(albumentations_augs)
    cropped_image, _ = albument(image, mask)
    # Use skimage greycoprops https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_glcm.html
    # https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.graycoprops
    glcm_sk = graycomatrix(
        cropped_image, distances=[1], angles=[0], normed=True, symmetric=True
    )
    glcm = glcm_sk[:, :, 0, 0]
    # Prepare coefficients
    i_indexes = np.zeros_like(glcm)
    j_indexes = np.zeros_like(glcm)
    i_means = np.zeros_like(glcm)
    j_means = np.zeros_like(glcm)
    p_x_vector = np.zeros_like(glcm)
    p_y_vector = np.zeros_like(glcm)
    for i in range(glcm.shape[0]):
        i_indexes[i, :] = i
        i_means[i, :] = np.mean(glcm[i, :])
        p_x_vector[i, :] = np.sum(glcm[i, :])
    for j in range(glcm.shape[1]):
        j_indexes[:, j] = j
        j_means[:, j] = np.mean(glcm[:, j])
        p_y_vector[:, j] = np.sum(glcm[:, j])
    glcm_mean = np.mean(glcm)
    per_k_p_x_plus_y = []
    for k in range(2, glcm.shape[0] * 2 + 1):
        per_k_p_x_plus_y.append(np.sum(np.fliplr(glcm).diagonal(glcm.shape[0] + 1 - k)))
    per_k_p_x_plus_y = np.asarray(per_k_p_x_plus_y)
    per_k_p_x_minus_y = []
    per_k_p_x_minus_y.append(glcm.diagonal(0).sum())
    for k in range(1, glcm.shape[0]):
        per_k_p_x_minus_y.append(glcm.diagonal(k).sum() + glcm.diagonal(-k).sum())
    per_k_p_x_minus_y = np.asarray(per_k_p_x_minus_y)
    k_vector = np.arange(2, 2 * glcm.shape[0] + 1, 1)
    HX = p_x_vector[:, 0].sum()
    HY = p_y_vector[0, :].sum()
    HXY1 = -(glcm * np.log(p_x_vector * p_y_vector + 1e-6)).sum()
    HXY2 = -(p_x_vector * p_y_vector * np.log(p_x_vector * p_y_vector + 1e-6)).sum()

    # Compute features
    T1 = (glcm ** 2).sum()  # ASM
    T2 = -(glcm * np.log(glcm + 1e-6)).sum()  # Entropy
    T3 = (np.abs(i_indexes - j_indexes) * glcm).sum()  # Dissimilarity
    T4 = ((np.arange(0, glcm.shape[1], 1) ** 2) * per_k_p_x_minus_y).sum()  # Contrast
    T5 = ((1 / (1 + np.abs(i_indexes - j_indexes))) * glcm).sum()  # Inverse difference
    T6 = ((1 / (1 + ((i_indexes - j_indexes) ** 2))) * glcm).sum()  # IDM
    T7 = graycoprops(glcm_sk, "correlation")[0, 0]  # Correlation
    T8 = (i_indexes * j_indexes * glcm).sum()  # Autocorrelation
    T9 = (
            ((i_indexes + j_indexes - i_means - j_means) ** 3) * glcm
    ).sum()  # Cluster shade
    T10 = (
            ((i_indexes + j_indexes - i_means - j_means) ** 4) * glcm
    ).sum()  # Cluster prominence
    T11 = np.max(glcm)  # Maximum probability
    T12 = (((i_indexes - glcm_mean) ** 2) * glcm).sum()  # Variance
    T13 = (k_vector * per_k_p_x_plus_y).sum()  # Sum average
    T15 = -(per_k_p_x_plus_y * np.log(per_k_p_x_plus_y + 1e-6)).sum()  # Sum entropy
    T14 = (((k_vector - T15) ** 2) * per_k_p_x_plus_y).sum()  # Sum variance
    T17 = -(
            per_k_p_x_minus_y * np.log(per_k_p_x_minus_y + 1e-6)
    ).sum()  # Difference entropy
    T16 = (
            ((i_indexes[:, 0] - T17) ** 2) * per_k_p_x_minus_y
    ).sum()  # Difference variance
    T18 = (T2 - HXY1) / np.max([HX, HY])  # IMC1
    T19 = np.sqrt((1 - np.exp(-2 * (HXY2 - T2))))  # IMC2
    T21 = (
            glcm / (1 + np.abs(i_indexes - j_indexes) / (glcm.shape[0] ** 2))
    ).sum()  # INN
    T22 = (
            glcm / (1 + ((i_indexes - j_indexes) ** 2) / (glcm.shape[0] ** 2))
    ).sum()  # IDN
    return (
        T1,
        T2,
        T3,
        T4,
        T5,
        T6,
        T7,
        T8,
        T9,
        T10,
        T11,
        T12,
        T13,
        T14,
        T15,
        T16,
        T17,
        T18,
        T19,
        T21,
        T22,
    )


def red_prop_features(image_rgb, mask):
    if np.max(image_rgb) > 1:
        image = cv2.normalize(
            image_rgb,
            None,
            alpha=0,
            beta=1,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_32F,
        )
    else:
        image = image_rgb
    if np.max(mask) > 1:
        mask = mask / 255
    if len(mask.shape) >= 3:
        mask = mask[:, :, 0]

    mask = mask.astype('bool')

    r_vals = image[:, :, 0][mask]
    g_vals = image[:, :, 1][mask]
    b_vals = image[:, :, 2][mask]
    r_sum = np.sum(r_vals)
    g_sum = np.sum(g_vals)
    b_sum = np.sum(b_vals)
    c1 = r_sum / g_sum
    c2 = r_sum / b_sum
    c3 = r_sum / (r_sum + g_sum + b_sum)  # chromacity
    # c4 = []  # Not sure if that's correct - the equation is quite enigmatic
    # c5 = []
    # for r_val, g_val, b_val in zip(r_vals, g_vals, b_vals):
    #     c4.append(r_val / (np.sqrt(g_val ** 2 + b_val ** 2) + 1e-5))
    #     c5.append(1 - (np.min([g_val, b_val]) / (r_val + 1e-5)))
    # , np.mean(c4), np.mean(c5)
    return c1, c2, c3


def rgb_hsv_means(image_rgb, mask):
    if np.max(image_rgb) > 1:
        image = cv2.normalize(
            image_rgb,
            None,
            alpha=0,
            beta=1,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_32F,
        )
    else:
        image = image_rgb
    image_hsv = hsv2rgb(deepcopy(image))

    if len(mask.shape) >= 3:
        mask = mask[:, :, 0]

    mask = mask.astype('bool')

    r_vals = image[:, :, 0][mask]
    g_vals = image[:, :, 1][mask]
    b_vals = image[:, :, 2][mask]
    h_vals = image_hsv[:, :, 0][mask]
    s_vals = image_hsv[:, :, 1][mask]
    v_vals = image_hsv[:, :, 2][mask]
    r_mean = np.mean(r_vals)
    g_mean = np.mean(g_vals)
    b_mean = np.mean(b_vals)
    h_mean = np.mean(h_vals)
    s_mean = np.mean(s_vals)
    v_mean = np.mean(v_vals)
    return r_mean, g_mean, b_mean, h_mean, s_mean, v_mean


def textural_features_mult_images(images_rgb, masks):
    images_rgb = np.asarray(images_rgb)
    images = np.zeros(images_rgb.shape[:-1])
    for k in range(images_rgb.shape[0]):
        images[k] = cv2.cvtColor(images_rgb[k], cv2.COLOR_RGB2GRAY)
    x_mins = np.zeros(images_rgb.shape[0], dtype=np.int_) - 1
    x_maxs = np.zeros(images_rgb.shape[0], dtype=np.int_) - 1
    y_mins = np.zeros(images_rgb.shape[0], dtype=np.int_) - 1
    y_maxs = np.zeros(images_rgb.shape[0], dtype=np.int_) - 1
    for i in range(masks.shape[1]):
        in_mask = np.any(masks[:, i, :], axis=1)
        y_mins = np.where(np.logical_and(in_mask, y_mins == -1), i, y_mins)
        y_maxs = np.where(in_mask, i, y_maxs)
    for j in range(masks.shape[2]):
        in_mask = np.any(masks[:, :, j], axis=1)
        x_mins = np.where(np.logical_and(in_mask, x_mins == -1), j, x_mins)
        x_maxs = np.where(in_mask, j, x_maxs)
    cropped_images = []
    for k, (image, mask, x_min, y_min, x_max, y_max) in enumerate(
        zip(images, masks, x_mins, y_mins, x_maxs, y_maxs)
    ):
        albumentations_augs = []
        albumentations_augs.append(A.Crop(x_min, y_min, x_max, y_max, p=1.0))
        albumentations_augs = A.Compose(albumentations_augs)
        albument = Albument(albumentations_augs)
        cropped_image, _ = albument(image, mask)
        cropped_images.append(cropped_image)
    # Use skimage greycoprops https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_glcm.html
    # https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.graycoprops
    glcms_sk = []
    glcms = []
    for image in cropped_images:
        glcms_sk.append(
            graycomatrix(
                image.astype("uint8"),
                distances=[1],
                angles=[0],
                normed=True,
                symmetric=True,
            )
        )
        glcms.append(glcms_sk[-1][:, :, 0, 0])
    glcms_sk = np.asarray(glcms_sk)
    glcms = np.asarray(glcms)

    # Prepare coefficients
    i_indexes = np.zeros_like(glcms[0])
    j_indexes = np.zeros_like(glcms[0])
    i_means = np.zeros_like(glcms)
    j_means = np.zeros_like(glcms)
    p_x_vector = np.zeros_like(glcms)
    p_y_vector = np.zeros_like(glcms)
    for i in range(glcms.shape[1]):
        i_indexes[i, :] = i
        i_means[:, i, :] = np.repeat(
            np.mean(glcms[:, i, :], axis=1)[:, np.newaxis], 256, axis=1
        )
        p_x_vector[:, i, :] = np.repeat(
            np.sum(glcms[:, i, :], axis=1)[:, np.newaxis], 256, axis=1
        )
    for j in range(glcms.shape[2]):
        j_indexes[:, j] = j
        j_means[:, :, j] = np.repeat(
            np.mean(glcms[:, :, j], axis=1)[:, np.newaxis], 256, axis=1
        )
        p_y_vector[:, :, j] = np.repeat(
            np.sum(glcms[:, :, j], axis=1)[:, np.newaxis], 256, axis=1
        )
    glcm_mean = np.repeat(
        np.repeat(np.mean(glcms, axis=(1, 2))[:, np.newaxis], 256, axis=1)[
            :, :, np.newaxis
        ],
        256,
        axis=2,
    )
    per_k_p_x_plus_y = np.zeros(glcms.shape[0], dtype="object")
    for idx, glcm in enumerate(glcms):
        per_k_p_x_plus_y[idx] = np.zeros(glcms.shape[1] * 2 - 1)
        for k in range(2, glcms.shape[1] * 2 + 1):
            per_k_p_x_plus_y[idx][k - 2] = np.sum(
                np.fliplr(glcm).diagonal(glcm.shape[0] + 1 - k)
            )
    per_k_p_x_plus_y = np.array(list(per_k_p_x_plus_y))
    per_k_p_x_minus_y = np.zeros(glcms.shape[0], dtype="object")
    for idx, glcm in enumerate(glcms):
        per_k_p_x_minus_y[idx] = np.zeros(glcm.shape[0])
        per_k_p_x_minus_y[idx][0] = glcm.diagonal(0).sum()
        for k in range(1, glcm.shape[0]):
            per_k_p_x_minus_y[idx][k] = glcm.diagonal(k).sum() + glcm.diagonal(-k).sum()
    per_k_p_x_minus_y = np.array(list(per_k_p_x_minus_y))
    k_vector = np.arange(2, 2 * glcms.shape[1] + 1, 1)
    HX = p_x_vector[:, :, 0].sum(axis=1)
    HY = p_y_vector[:, 0, :].sum(axis=1)
    HXY1 = -(glcms * np.log(p_x_vector * p_y_vector + 1e-6)).sum(axis=(1, 2))
    HXY2 = -(p_x_vector * p_y_vector * np.log(p_x_vector * p_y_vector + 1e-6)).sum(
        axis=(1, 2)
    )

    # Compute features
    T1 = (glcms ** 2).sum(axis=(1, 2))  # ASM
    T2 = -(glcms * np.log(glcms + 1e-6)).sum(axis=(1, 2))  # Entropy
    T3 = (np.abs(i_indexes - j_indexes) * glcms).sum(axis=(1, 2))  # Dissimilarity
    T4 = ((np.arange(0, glcms.shape[1], 1) ** 2) * per_k_p_x_minus_y).sum(
        axis=1
    )  # Contrast
    T5 = ((1 / (1 + np.abs(i_indexes - j_indexes))) * glcms).sum(
        axis=(1, 2)
    )  # Inverse difference
    T6 = ((1 / (1 + ((i_indexes - j_indexes) ** 2))) * glcms).sum(axis=(1, 2))  # IDM
    T7 = []  # Correlation
    for glcm_sk in glcms_sk:
        T7.append(graycoprops(glcm_sk, "correlation")[0, 0])
    T7 = np.asarray(T7)
    T8 = (i_indexes * j_indexes * glcms).sum(axis=(1, 2))  # Autocorrelation
    T9 = (((i_indexes + j_indexes - i_means - j_means) ** 3) * glcms).sum(
        axis=(1, 2)
    )  # Cluster shade
    T10 = (((i_indexes + j_indexes - i_means - j_means) ** 4) * glcms).sum(
        axis=(1, 2)
    )  # Cluster prominence
    T11 = np.max(glcms, axis=(1, 2))  # Maximum probability
    T12 = (((i_indexes - glcm_mean) ** 2) * glcms).sum(axis=(1, 2))  # Variance
    T13 = (k_vector * per_k_p_x_plus_y).sum(axis=1)  # Sum average
    T15 = -(per_k_p_x_plus_y * np.log(per_k_p_x_plus_y + 1e-6)).sum(
        axis=1
    )  # Sum entropy
    T14 = (
        ((k_vector - np.repeat(T15[:, np.newaxis], len(k_vector), axis=1)) ** 2)
        * per_k_p_x_plus_y
    ).sum(
        axis=1
    )  # Sum variance
    T17 = -(per_k_p_x_minus_y * np.log(per_k_p_x_minus_y + 1e-6)).sum(
        axis=1
    )  # Difference entropy
    T16 = (
        ((i_indexes[:, 0] - np.repeat(T17[:, np.newaxis], 256, axis=1)) ** 2)
        * per_k_p_x_minus_y
    ).sum(
        axis=1
    )  # Difference variance
    T18 = (T2 - HXY1) / np.max([HX, HY])  # IMC1
    T19 = np.sqrt((1 - np.exp(-2 * (HXY2 - T2))))  # IMC2
    #     if two_max_vals[1] == -1:
    #         T20 = np.sqrt(two_max_vals[0]) # Maximal correlation coefficient
    #     else:
    #         T20 = np.sqrt(two_max_vals[1]) # Maximal correlation coefficient
    T21 = (glcms / (1 + np.abs(i_indexes - j_indexes) / (glcms.shape[1] ** 2))).sum(
        axis=(1, 2)
    )  # INN
    T22 = (glcms / (1 + ((i_indexes - j_indexes) ** 2) / (glcms.shape[1] ** 2))).sum(
        axis=(1, 2)
    )  # IDN
    return (
        T1,
        T2,
        T3,
        T4,
        T5,
        T6,
        T7,
        T8,
        T9,
        T10,
        T11,
        T12,
        T13,
        T14,
        T15,
        T16,
        T17,
        T18,
        T19,
        T21,
        T22,
    )

def red_prop_features_mult_images(images_rgb, masks):
    if np.max(images_rgb[0]) > 1:
        for idx, image_rgb in enumerate(images_rgb):
                images_rgb[idx] = cv2.normalize(
                    image_rgb,
                    None,
                    alpha=0,
                    beta=1,
                    norm_type=cv2.NORM_MINMAX,
                    dtype=cv2.CV_32F,
                )
    if np.max(masks[0]) > 1:
        masks = masks / 255
    if len(masks.shape) >= 4:
        masks = masks[:, :, :, 0]

    masks = masks.astype('bool')

    r_vals = images_rgb[:, :, :, 0][masks]
    g_vals = images_rgb[:, :, :, 1][masks]
    b_vals = images_rgb[:, :, :, 2][masks]
    r_sum = r_vals.sum(axis=(1, 2))
    g_sum = g_vals.sum(axis=(1, 2))
    b_sum = b_vals.sum(axis=(1, 2))
    c1 = r_sum / g_sum
    c2 = r_sum / b_sum
    c3 = r_sum / (r_sum + g_sum + b_sum)  # chromacity
    # c4 = []  # Not sure if that's correct - the equation is quite enigmatic
    # c5 = []
    # for r_val, g_val, b_val in zip(r_vals, g_vals, b_vals):
    #     c4.append(r_val / (np.sqrt(g_val ** 2 + b_val ** 2) + 1e-5))
    #     c5.append(1 - (np.min([g_val, b_val]) / (r_val + 1e-5)))
    # , np.mean(c4), np.mean(c5)
    return c1, c2, c3


def rgb_hsv_means_mult_images(images_rgb, masks):
    if np.max(images_rgb[0]) > 1:
        for idx, image_rgb in enumerate(images_rgb):
            images_rgb[idx] = cv2.normalize(
                image_rgb,
                None,
                alpha=0,
                beta=1,
                norm_type=cv2.NORM_MINMAX,
                dtype=cv2.CV_32F,
            )
    if np.max(masks[0]) > 1:
        masks = masks / 255
    if len(masks.shape) >= 4:
        masks = masks[:, :, :, 0]

    masks = masks.astype('bool')

    image_hsv = np.array([hsv2rgb(deepcopy(image)) for image in images_rgb])

    r_vals = images_rgb[:, :, :, 0][masks]
    g_vals = images_rgb[:, :, :, 1][masks]
    b_vals = images_rgb[:, :, :, 2][masks]
    h_vals = image_hsv[:, :, :, 0][masks]
    s_vals = image_hsv[:, :, :, 1][masks]
    v_vals = image_hsv[:, :, :, 2][masks]
    r_mean = np.mean(r_vals)
    g_mean = np.mean(g_vals)
    b_mean = np.mean(b_vals)
    h_mean = np.mean(h_vals)
    s_mean = np.mean(s_vals)
    v_mean = np.mean(v_vals)
    return r_mean, g_mean, b_mean, h_mean, s_mean, v_mean
