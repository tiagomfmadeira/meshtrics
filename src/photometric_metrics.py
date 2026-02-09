import sys
import warnings
from enum import Enum
from io import BytesIO

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyrender
import seaborn as sns
import trimesh
from psnr_hvsm import hvs_mse, hvsm_mse
from pyrender.constants import RenderFlags
from scipy import signal
from skimage.filters.rank import entropy


class Filter(Enum):
    UNIFORM = 0
    GAUSSIAN = 1


def sim_view_point(mesh, cam_matrix, intrinsics, width, height):
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[2.5, 2.5, 2.5])

    camera = pyrender.IntrinsicsCamera(fx=intrinsics[0, 0], fy=intrinsics[1, 1],
                                       cx=intrinsics[0, 2], cy=intrinsics[1, 2],
                                       znear=0.001, zfar=10000)
    scene.add(camera, pose=cam_matrix)

    r = pyrender.OffscreenRenderer(width, height)

    if isinstance(mesh, trimesh.Trimesh):
        scene.add(pyrender.Mesh.from_trimesh(mesh))
        # pyrender.Viewer(scene)
        # color, depth
        return (r.render(scene, flags=RenderFlags.RGBA | RenderFlags.FLAT))

    elif isinstance(mesh, trimesh.Scene):
        for geometry in mesh.geometry.values():
            scene.add(pyrender.Mesh.from_trimesh(geometry))
        # pyrender.Viewer(scene)
        # color, depth
        return (r.render(scene, flags=RenderFlags.RGBA))

    else:
        print("\nERROR! Cannot simulate view with given mesh object.")
        print("Accepted is trimesh.Trimesh or trimesh.Scene")
        print("Called with:")
        print(type(mesh).__module__ + '.' + type(mesh).__qualname__)
        return None


# No reference
def get_entropy(image, neighborhood, region_name, mask=None, output_path='.'):
    entropy_data = entropy(image, neighborhood, mask=mask)

    print("\nImage entropy for region = {:.5f}".format(entropy_data.sum() / mask.sum()))

    # Heatmap figure
    fig = plt.figure()
    plt.imshow(entropy_data, cmap='gist_yarg')
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path + '/noref_output/' + region_name + '_entropy_heatmap.png', bbox_inches='tight',
                format='png')
    fig.clf()
    plt.close()


# Full reference
def matrix_from_vecs(rvecs, tvecs):
    estimated_r, jacobian = cv2.Rodrigues(rvecs)

    fg_cam_matrix = np.zeros((4, 4))
    fg_cam_matrix[3, 3] = 1
    fg_cam_matrix[:-1, -1] = tvecs.reshape(3, )
    fg_cam_matrix[:3, :3] = estimated_r

    fg_cam_matrix[1, :] = -fg_cam_matrix[1, :]
    fg_cam_matrix[2, :] = -fg_cam_matrix[2, :]
    fg_cam_matrix = np.linalg.inv(fg_cam_matrix)

    return fg_cam_matrix


def solve_pnp(points_3d, points_2d, intrinsics, dist=None):
    retval, rvecs, tvecs, inliers = cv2.solvePnPRansac(points_3d, points_2d, intrinsics, dist)

    if inliers is not None:
        points_3d = np.array([points_3d[idx][0] for idx in inliers])
        points_2d = np.array([points_2d[idx][0] for idx in inliers])
    else:
        return None

    rvecs, tvecs = cv2.solvePnPRefineLM(points_3d, points_2d, intrinsics, dist, rvecs, tvecs, (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 10000, sys.float_info.epsilon))

    fg_cam_matrix = matrix_from_vecs(rvecs, tvecs)

    return fg_cam_matrix


def initial_check(target, source):
    assert target.shape == source.shape, "Supplied images have different sizes " + str(target.shape) + " and " + str(
        source.shape)
    if target.dtype != source.dtype:
        msg = "Supplied images have different dtypes " + str(target.dtype) + " and " + str(source.dtype)
        warnings.warn(msg)

    return target.astype(np.float64), source.astype(np.float64)


def mse(target, source, mask=None):
    target, source = initial_check(target, source)
    mse_map = (target.astype(np.float64) - source.astype(np.float64)) ** 2
    mask_3chan = np.ones_like(target)
    if mask is not None:
        mask_3chan[:, :, 0] = mask
        mask_3chan[:, :, 1] = mask
        mask_3chan[:, :, 2] = mask

    # Save heatmap figure to buffer
    h, w, c = target.shape
    px = 1 / plt.rcParams['figure.dpi']
    fig = plt.figure(figsize=(w * px, h * px))
    sns.heatmap(((mse_map[:, :, 0] + mse_map[:, :, 1] + mse_map[:, :, 2]) / 3) * mask, cmap='gist_yarg', robust=False,
                cbar=False, mask=np.logical_not(mask))
    plt.axis('off')
    buffer = BytesIO()
    plt.savefig(buffer, bbox_inches='tight', format='png')
    fig.clf()
    plt.close()

    return (mask_3chan * mse_map).sum() / mask_3chan.sum(), buffer


def rmse(target, source, mask=None):
    target, source = initial_check(target, source)
    result, fig = mse(target, source, mask)
    return np.sqrt(result), fig


def psnr_hvs(target, source, mode="hvs", mask=None):
    target, source = initial_check(target, source)

    if mode.lower() == "hvs":
        hvs_scores = hvs_mse(target, source)
    elif mode.lower() == "hvsm":
        hvs_scores = hvsm_mse(target, source)
    else:
        print("Invalid mode! Please choose HVS or HVSM!")
        return None

    if mask is None:
        hvs_scores_mask = np.ones(hvs_scores.shape)
    else:
        hvs_scores_mask = np.zeros(hvs_scores.shape)
        height, width = hvs_scores_mask.shape
        tile_size = 8
        for i in range(height):
            for j in range(width):
                if np.all(mask[i * tile_size:(i + 1) * tile_size, j * tile_size:(j + 1) * tile_size]):
                    hvs_scores_mask[i, j] = 1

    hvs_mean = (hvs_scores_mask * hvs_scores).sum() / hvs_scores_mask.sum()

    # Save heatmap figure to buffer
    h, w = target.shape
    px = 1 / plt.rcParams['figure.dpi']
    fig = plt.figure(figsize=(w * px, h * px))
    sns.heatmap(hvs_scores_mask * hvs_scores, cmap='gist_yarg', robust=False, cbar=False,
                mask=np.logical_not(hvs_scores_mask))
    plt.axis('off')
    buffer = BytesIO()
    plt.savefig(buffer, bbox_inches='tight', format='png')
    fig.clf()
    plt.close()

    if hvs_mean == 0:
        result = 100
    else:
        result = min(10.0 * np.log10(1.0 / hvs_mean), 100.0)

    return result, buffer


def get_sums(target, source, win, mode='same'):
    mu1, mu2 = (signal.convolve2d(target, np.rot90(win, 2), mode=mode),
                signal.convolve2d(source, np.rot90(win, 2), mode=mode))
    return mu1 * mu1, mu2 * mu2, mu1 * mu2


def get_sigmas(target, source, win, mode='same', **kwargs):
    if 'sums' in kwargs:
        target_sum_sq, source_sum_sq, target_source_sum_mul = kwargs['sums']
    else:
        target_sum_sq, source_sum_sq, target_source_sum_mul = get_sums(target, source, win, mode)

    return signal.convolve2d(target * target, np.rot90(win, 2), mode=mode) - target_sum_sq, \
           signal.convolve2d(source * source, np.rot90(win, 2), mode=mode) - source_sum_sq, \
           signal.convolve2d(target * source, np.rot90(win, 2), mode=mode) - target_source_sum_mul


def fspecial(fltr, ws, **kwargs):
    if fltr == Filter.UNIFORM:
        return np.ones((ws, ws)) / ws ** 2
    elif fltr == Filter.GAUSSIAN:
        x, y = np.mgrid[-ws // 2 + 1:ws // 2 + 1, -ws // 2 + 1:ws // 2 + 1]
        g = np.exp(-((x ** 2 + y ** 2) / (2.0 * kwargs['sigma'] ** 2)))
        g[g < np.finfo(g.dtype).eps * g.max()] = 0
        assert g.shape == (ws, ws)
        den = g.sum()
        if den != 0:
            g /= den
        return g
    return None


def ssim_single(target, source, mask, ws, c1, c2, fltr_specs, mode):
    win = fspecial(**fltr_specs)

    target_sum_sq, source_sum_sq, target_source_sum_mul = get_sums(target, source, win, mode)
    target_sigma_sq, source_sigma_sq, target_source_sigma = get_sigmas(target, source, win, mode, sums=(
        target_sum_sq, source_sum_sq, target_source_sum_mul))

    assert c1 > 0
    assert c2 > 0

    ssim_map = ((2 * target_source_sum_mul + c1) * (2 * target_source_sigma + c2)) / (
            (target_sum_sq + source_sum_sq + c1) * (target_sigma_sq + source_sigma_sq + c2))
    cs_map = (2 * target_source_sigma + c2) / (target_sigma_sq + source_sigma_sq + c2)

    # SSIM can be negative (-1 to 1). Normalize value to 0-1 so heatmap makes sense...
    ssim_map = (ssim_map + 1) / 2

    if mask is None:
        return np.mean(ssim_map), np.mean(cs_map), ssim_map, None
    else:
        s = int((ws - 1) / 2)
        mask = mask[s:-s, s:-s]
        kernel = np.ones((ws, ws), np.uint8)
        mask = cv2.erode(np.float32(mask), kernel, iterations=1)
        return (mask * ssim_map).sum() / mask.sum(), (mask * cs_map).sum() / mask.sum(), mask * ssim_map, mask


def ssim(target, source, mask=None, ws=11, k1=0.01, k2=0.03, max=None, fltr_specs=None, mode='valid'):
    if max is None:
        max = np.iinfo(target.dtype).max

    target, source = initial_check(target, source)

    if fltr_specs is None:
        fltr_specs = dict(fltr=Filter.UNIFORM, ws=ws)

    c1 = (k1 * max) ** 2
    c2 = (k2 * max) ** 2

    ssims = []
    css = []
    heatmaps = []

    if mask is None:
        mask = np.ones_like(target[:, :, 0])

    for i in range(target.shape[2]):
        ssim, cs, heatmap, show_mask = ssim_single(target[:, :, i], source[:, :, i], mask, ws, c1, c2, fltr_specs, mode)
        ssims.append(ssim)
        css.append(cs)
        heatmaps.append(heatmap)

    show_heatmap = np.zeros(heatmaps[0].shape)
    for heatmap in heatmaps:
        show_heatmap = show_heatmap + heatmap

    # Save heatmap figure to buffer
    show_mask = np.logical_not(show_mask)
    h, w, c = target.shape
    px = 1 / plt.rcParams['figure.dpi']
    fig = plt.figure(figsize=(w * px, h * px))
    sns.heatmap(show_heatmap / len(heatmaps), cmap='gist_gray', robust=False, cbar=False, mask=show_mask)
    plt.axis('off')
    buffer = BytesIO()
    plt.savefig(buffer, bbox_inches='tight', format='png')
    fig.clf()
    plt.close()

    return np.mean(ssims), np.mean(css), buffer


def vifp_single(target, source, sigma_nsq, mask=None):
    EPS = 1e-10
    num = 0.0
    den = 0.0

    for scale in range(1, 5):
        N = 2.0 ** (4 - scale + 1) + 1
        win = fspecial(Filter.GAUSSIAN, ws=N, sigma=N / 5)

        if scale > 1:
            target = signal.convolve2d(target, np.rot90(win, 2), mode='valid')[::2, ::2]
            source = signal.convolve2d(source, np.rot90(win, 2), mode='valid')[::2, ::2]

        target_sum_sq, source_sum_sq, target_source_sum_mul = get_sums(target, source, win, mode='valid')
        target_sigma_sq, source_sigma_sq, target_source_sigma = get_sigmas(target, source, win, mode='valid', sums=(
            target_sum_sq, source_sum_sq, target_source_sum_mul))

        target_sigma_sq[target_sigma_sq < 0] = 0
        source_sigma_sq[source_sigma_sq < 0] = 0

        g = target_source_sigma / (target_sigma_sq + EPS)
        sv_sq = source_sigma_sq - g * target_source_sigma

        g[target_sigma_sq < EPS] = 0
        sv_sq[target_sigma_sq < EPS] = source_sigma_sq[target_sigma_sq < EPS]
        target_sigma_sq[target_sigma_sq < EPS] = 0

        g[source_sigma_sq < EPS] = 0
        sv_sq[source_sigma_sq < EPS] = 0

        sv_sq[g < 0] = source_sigma_sq[g < 0]
        g[g < 0] = 0
        sv_sq[sv_sq <= EPS] = EPS

        if mask is None:
            num += np.sum(np.log10(1.0 + (g ** 2.) * target_sigma_sq / (sv_sq + sigma_nsq)))
            den += np.sum(np.log10(1.0 + target_sigma_sq / sigma_nsq))
        else:
            s = int((len(win) - 1) / 2)

            if scale > 1:
                mask = mask[s:-s, s:-s][::2, ::2]
                kernel = np.ones(win.shape, np.uint8)
                mask = cv2.erode(np.float32(mask), kernel, iterations=1)

            tmp_mask = mask[s:-s, s:-s]
            kernel = np.ones(win.shape, np.uint8)
            tmp_mask = cv2.erode(np.float32(tmp_mask), kernel, iterations=1)

            num += np.sum(tmp_mask * (np.log10(1.0 + (g ** 2.) * target_sigma_sq / (sv_sq + sigma_nsq))))
            den += np.sum(tmp_mask * (np.log10(1.0 + target_sigma_sq / sigma_nsq)))

            ###########################################################################
            # Combine and store values for heatmap in all scales
            if scale == 1:
                tmp_num = tmp_mask * (np.log10(1.0 + (g ** 2.) * target_sigma_sq / (sv_sq + sigma_nsq)))
                tmp_den = tmp_mask * (np.log10(1.0 + target_sigma_sq / sigma_nsq))

            elif scale == 2:
                tmp_num[s:-s, s:-s][::2, ::2] = np.add(tmp_num[s:-s, s:-s][::2, ::2], tmp_mask * (
                    np.log10(1.0 + (g ** 2.) * target_sigma_sq / (sv_sq + sigma_nsq))))
                tmp_den[s:-s, s:-s][::2, ::2] = np.add(tmp_den[s:-s, s:-s][::2, ::2],
                                                       tmp_mask * (np.log10(1.0 + target_sigma_sq / sigma_nsq)))
            elif scale == 3:
                i = s * (2 ** 1)

                tmp_num[i:-i, i:-i][::2, ::2][s:-s, s:-s][::2, ::2] = np.add(
                    tmp_num[i:-i, i:-i][::2, ::2][s:-s, s:-s][::2, ::2],
                    tmp_mask * (np.log10(1.0 + (g ** 2.) * target_sigma_sq / (sv_sq + sigma_nsq))))
                tmp_den[i:-i, i:-i][::2, ::2][s:-s, s:-s][::2, ::2] = np.add(
                    tmp_den[i:-i, i:-i][::2, ::2][s:-s, s:-s][::2, ::2],
                    tmp_mask * (np.log10(1.0 + target_sigma_sq / sigma_nsq)))
            elif scale == 4:
                j = s * (2 ** 1)
                i = s * (2 ** 2)

                tmp_num[i:-i, i:-i][::2, ::2][j:-j, j:-j][::2, ::2][s:-s, s:-s][::2, ::2] = np.add(
                    tmp_num[i:-i, i:-i][::2, ::2][j:-j, j:-j][::2, ::2][s:-s, s:-s][::2, ::2],
                    tmp_mask * (np.log10(1.0 + (g ** 2.) * target_sigma_sq / (sv_sq + sigma_nsq))))
                tmp_den[i:-i, i:-i][::2, ::2][j:-j, j:-j][::2, ::2][s:-s, s:-s][::2, ::2] = np.add(
                    tmp_num[i:-i, i:-i][::2, ::2][j:-j, j:-j][::2, ::2][s:-s, s:-s][::2, ::2],
                    tmp_mask * (np.log10(1.0 + target_sigma_sq / sigma_nsq)))

    return num / den, np.divide(tmp_num, tmp_den)


def vifp(target, source, sigma_nsq=2, mask=None):
    np.seterr(divide='ignore', invalid='ignore')
    target, source = initial_check(target, source)

    vifps = []
    heatmaps = []

    if mask is None:
        mask = np.ones_like(target[:, :, 0])

    for i in range(target.shape[2]):
        vifp, heatmap = vifp_single(target[:, :, i], source[:, :, i], sigma_nsq, mask)
        vifps.append(vifp)
        heatmaps.append(heatmap)

    # Average heatmap for 3 channels
    show_heatmap = np.zeros(heatmaps[0].shape)
    for heatmap in heatmaps:
        show_heatmap = show_heatmap + heatmap

    show_heatmap = show_heatmap / len(heatmaps)

    # Save heatmap figure to buffer
    h, w, c = target.shape
    px = 1 / plt.rcParams['figure.dpi']
    fig = plt.figure(figsize=(w * px, h * px))
    sns.heatmap(show_heatmap, cmap='gist_gray', robust=False, cbar=False, mask=np.logical_not(mask[8:-8, 8:-8]))
    plt.axis('off')
    buffer = BytesIO()
    plt.savefig(buffer, bbox_inches='tight', format='png')
    fig.clf()
    plt.close()

    return np.mean(vifps), buffer


def compare_images(img1, img2, cam_name, mask=None, output_path='.'):
    # RMSE
    rmse_result, buffer = rmse(img1[:, :, :3], img2[:, :, :3], mask=mask)
    rmse_result = 1 - (rmse_result / 255)
    if output_path:
        buffer.seek(0)
        raw = buffer.getvalue()
        buffer.close()
        fig_array = np.fromstring(raw, dtype='uint8')
        img = cv2.imdecode(fig_array, cv2.IMREAD_UNCHANGED)
        cv2.imwrite(output_path + '/' + cam_name + '_MSE_heatmap.png', img)
    # print("\nRMSE = {:.5f}".format(rmse_result))

    # PSNR_HVS_M
    # Convert RGB to YUV colorspace and get only the luma component (Y)
    y1, u1, v1 = cv2.split(cv2.cvtColor(img1, cv2.COLOR_RGB2YUV))
    y2, u2, v2 = cv2.split(cv2.cvtColor(img2, cv2.COLOR_RGB2YUV))
    y1 = y1.astype(float) / 255
    y2 = y2.astype(float) / 255
    psnr_result, buffer = psnr_hvs(y1, y2, "hvsm", mask=mask)
    psnr_result = psnr_result / 100
    if output_path:
        buffer.seek(0)
        raw = buffer.getvalue()
        buffer.close()
        fig_array = np.fromstring(raw, dtype='uint8')
        img = cv2.imdecode(fig_array, cv2.IMREAD_UNCHANGED)
        cv2.imwrite(output_path + '/' + cam_name + '_PSNR_HVS_M_heatmap.png', img)
    # print("\nPSNR_HVS_M = {:.5f}".format(psnr_result))

    # SSIM
    ssim_result, cs, buffer = ssim(img1[:, :, :3], img2[:, :, :3], mask=mask)
    if output_path:
        buffer.seek(0)
        raw = buffer.getvalue()
        buffer.close()
        fig_array = np.fromstring(raw, dtype='uint8')
        img = cv2.imdecode(fig_array, cv2.IMREAD_UNCHANGED)
        cv2.imwrite(output_path + '/' + cam_name + '_SSIM_heatmap.png', img)
    # print("\nSSIM = {:.5f}".format(ssim_result))

    # VIFP
    vifp_result, buffer = vifp(img1[:, :, :3], img2[:, :, :3], mask=mask)
    if output_path:
        buffer.seek(0)
        raw = buffer.getvalue()
        buffer.close()
        fig_array = np.fromstring(raw, dtype='uint8')
        img = cv2.imdecode(fig_array, cv2.IMREAD_UNCHANGED)
        cv2.imwrite(output_path + '/' + cam_name + '_VIFP_heatmap.png', img)
    # print("\nVIFP = {:.5f}".format(vifp_result))

    return rmse_result, psnr_result, ssim_result, vifp_result
