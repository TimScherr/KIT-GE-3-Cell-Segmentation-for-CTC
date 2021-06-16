import json
import math
import numpy as np
import os
import tifffile as tiff

from copy import deepcopy
from pathlib import Path
from random import shuffle, random
from scipy.ndimage import gaussian_filter
import shutil
from skimage.measure import regionprops
from skimage.morphology import binary_closing, binary_opening
from skimage.transform import rescale

from segmentation.training.train_data_representations import distance_label_2d
from segmentation.utils.utils import get_nucleus_ids


def get_gt_frames(path_train_set):
    """ Get GT frames used (so that these frames are not used for STs in GT+ST setting)

    :param path_train_set: path to Cell Tracking Challenge training data sets.
        :type path_train_set: pathlib Path object
    :return: List of used GT frames
    """

    seg_gt_ids_01 = sorted((path_train_set / '01_GT' / 'SEG').glob('*.tif'))
    seg_gt_ids_02 = sorted((path_train_set / '02_GT' / 'SEG').glob('*.tif'))

    gt_frames_used = []

    for seg_gt_id in seg_gt_ids_01:
        if len(seg_gt_id.stem.split('_')) > 2:  # only slice annotated
            gt_frames_used.append("01_{}".format(seg_gt_id.stem.split('_')[2]))
        else:
            gt_frames_used.append("01_{}".format(seg_gt_id.stem.split('man_seg')[-1]))

    for seg_gt_id in seg_gt_ids_02:
        if len(seg_gt_id.stem.split('_')) > 2:  # only slice annotated
            gt_frames_used.append("02_{}".format(seg_gt_id.stem.split('_')[2]))
        else:
            gt_frames_used.append("02_{}".format(seg_gt_id.stem.split('man_seg')[-1]))

    return gt_frames_used


def close_st(seg_st, kernel_closing=np.ones((10, 10)), kernel_opening=np.ones((10, 10))):
    """ Morphological closing of STs.

    :param seg_st: Segmentation ST.
        :type seg_st:
    :param kernel_closing: Kernel for closing.
        :type kernel_closing:
    :param kernel_opening: Kernel for opening.
        :type kernel_opening:
    :return: Closed ST (with objects smaller than kernel_opening removed).
    """

    nucleus_ids = get_nucleus_ids(seg_st)
    hlabel = np.zeros(shape=seg_st.shape, dtype=seg_st.dtype)

    for nucleus_id in nucleus_ids:

        nucleus = seg_st == nucleus_id

        # Close nucleus gaps
        nucleus = binary_closing(nucleus, kernel_closing)

        # Remove small single not connected pixels
        nucleus = binary_opening(nucleus, kernel_opening)
        hlabel[nucleus] = nucleus_id.astype(seg_st.dtype)

    return hlabel


def generate_data(img, mask, tra_gt, search_radius, max_mal, crop_size, cell_type, mode, train_set, frame, min_area,
                  scale, path_train_sets, slice_idx=None):
    """ Calculate cell and neighbor distances and create crops.

    :param img: Image.
        :type img:
    :param mask: (Segmentation) Mask / label image (intensity coded).
        :type mask:
    :param tra_gt: Tracking ground truth (needed to evaluate if all cells are annotated).
        :type tra_gt:
    :param search_radius: Search radius for neighbor distances (affects computation time).
        :type search_radius: int
    :param max_mal: Maximum major axis length in training data (needed for cell distances).
        :type max_mal: int
    :param crop_size: Size of the created (square) crops.
        :type crop_size: int
    :param cell_type: Cell type (needed for filename).
        :type cell_type: str
    :param mode: Primary Track mode ('GT', 'ST', 'GT+ST', 'allGT', 'allST', 'allGT+allST') needed for filename.
        :type mode: str
    :param train_set: Set '01' or '02' (needed for filename).
        :type train_set: str
    :param frame: Frame of the time series (needed for filename).
        :type frame: str
    :param min_area: Minimum cell size in the data set (only partially visible cells are removed).
        :type min_area: int
    :param scale: Scale factor for downsampling.
        :type scale: float
    :param path_train_sets: Path of the created training sets.
        :type path_train_sets: pathlib Path objet.
    :param slice_idx: Slice index (for 3D data).
        :type slice_idx: int
    :return: None
    """

    # Calculate train data representations
    cell_dist, neighbor_dist = distance_label_2d(label=mask,
                                                 cell_radius=int(np.ceil(0.5 * max_mal)),
                                                 neighbor_radius=search_radius)

    # Adjust image dimensions for appropriate cropping
    img, mask, cell_dist, neighbor_dist, tra_gt = adjust_dimensions(crop_size, img, mask, cell_dist,  neighbor_dist,
                                                                    tra_gt)

    # Cropping
    nx = math.floor(img.shape[1] / crop_size)
    ny = math.floor(img.shape[0] / crop_size)
    for y in range(ny):
        for x in range(nx):

            img_crop, mask_crop, cell_dist_crop, neighbor_dist_crop, tra_gt_crop = get_crop(x, y, crop_size,
                                                                                            img, mask, cell_dist,
                                                                                            neighbor_dist, tra_gt)

            if slice_idx is not None:
                crop_name = '{}_{}_{}_{}_{:02d}_{:02d}_{:02d}.tif'.format(cell_type, mode, train_set,
                                                                          frame.split('.tif')[0], slice_idx, y, x)
            else:
                crop_name = '{}_{}_{}_{}_{:02d}_{:02d}.tif'.format(cell_type, mode, train_set,
                                                                   frame.split('.tif')[0], y, x)

            # Check cell number TRA/SEG
            tr_ids, mask_ids = get_nucleus_ids(tra_gt_crop), get_nucleus_ids(mask_crop)
            if np.sum(mask_crop[10:-10, 10:-10, 0] > 0) < min_area:  # only cell parts / no cell
                continue
            if len(mask_ids) == 1:  # neigbor may be cut from crop --> set dist to 0
                neighbor_dist_crop = np.zeros_like(neighbor_dist_crop)
            if np.sum(img_crop == 0) > (0.66 * img_crop.shape[0] * img_crop.shape[1]):  # almost background
                # For GOWT1 cells a lot of 0s are in the image
                if np.min(img_crop[:100, :100, ...]) == 0:
                    if np.sum(gaussian_filter(np.squeeze(img_crop), sigma=1) == 0) > (0.66 * img_crop.shape[0] * img_crop.shape[1]):
                        continue
                else:
                    continue
            if np.max(cell_dist_crop) < 0.8:
                continue

            # Remove only partially visible cells in mask for better comparison with tra_gt
            props_crop, n_part = regionprops(mask_crop), 0
            for cell in props_crop:
                if cell.area <= 0.1 * min_area and scale == 1:  # needed since tra_gt seeds are smaller
                    n_part += 1
            if (len(mask_ids) - n_part) >= len(tr_ids):  # A: all cells annotated
                crop_quality = 'A'
            elif (len(mask_ids) - n_part) >= 0.8 * len(tr_ids):  # >= 80% of the cells annotated
                crop_quality = 'B'
            elif (len(mask_ids) - n_part) >= 0.65 * len(tr_ids):  # >= 65% of the cells annotated
                crop_quality = 'C'
            else:  # not usable
                crop_quality = 'D'

            tiff.imsave(str(path_train_sets / "{}_{}".format(cell_type, mode) / crop_quality / ('img_' + crop_name)),
                        img_crop)
            tiff.imsave(str(path_train_sets / "{}_{}".format(cell_type, mode) / crop_quality / ('mask_' + crop_name)),
                        mask_crop)
            tiff.imsave(
                str(path_train_sets / "{}_{}".format(cell_type, mode) / crop_quality / ('dist_cell_' + crop_name)),
                cell_dist_crop)
            tiff.imsave(
                str(path_train_sets / "{}_{}".format(cell_type, mode) / crop_quality / ('dist_neighbor_' + crop_name)),
                neighbor_dist_crop)

    return None


def generate_data_st(img, mask, search_radius, max_mal, crop_size, cell_type, mode, train_set, frame, min_area,
                     scale, path_train_sets, slice_idx=None, running_crop_idx=0, max_crops=1000):
    """ Calculate cell and neighbor distances and create crops (for ST --> no TRA GT).

    :param img: Image.
        :type img:
    :param mask: (Segmentation) Mask / label image (intensity coded).
        :type mask:
    :param search_radius: Search radius for neighbor distances (affects computation time).
        :type search_radius: int
    :param max_mal: Maximum major axis length in training data (needed for cell distances).
        :type max_mal: int
    :param crop_size: Size of the created (square) crops.
        :type crop_size: int
    :param cell_type: Cell type (needed for filename).
        :type cell_type: str
    :param mode: Primary Track mode ('GT', 'ST', 'GT+ST', 'allGT', 'allST', 'allGT+allST') needed for filename.
        :type mode: str
    :param train_set: Set '01' or '02' (needed for filename).
        :type train_set: str
    :param frame: Frame of the time series (needed for filename).
        :type frame: str
    :param min_area: Minimum cell size in the data set (only partially visible cells are removed).
        :type min_area: int
    :param scale: Scale factor for downsampling.
        :type scale: float
    :param path_train_sets: Path of the created training sets.
        :type path_train_sets: pathlib Path objet.
    :param slice_idx: Slice index (for 3D data).
        :type slice_idx: int
    :param running_crop_idx: Crop index / amount crops created.
        :type running_crop_idx: int
    :param max_crops: Maximum amount of crops to create per data set.
        :type max_crops: int
    :return: None
    """

    # Calculate train data representations
    cell_dist, neighbor_dist = distance_label_2d(label=mask,
                                                 cell_radius=int(np.ceil(0.5 * max_mal)),
                                                 neighbor_radius=search_radius)

    # Adjust image dimensions for appropriate cropping
    img, mask, cell_dist, neighbor_dist = adjust_dimensions(crop_size, img, mask, cell_dist, neighbor_dist)

    # Cropping
    nx = math.floor(img.shape[1] / crop_size)
    ny = math.floor(img.shape[0] / crop_size)
    for y in range(ny):
        for x in range(nx):

            img_crop, mask_crop, cell_dist_crop, neighbor_dist_crop = get_crop(x, y, crop_size, img, mask, cell_dist,
                                                                               neighbor_dist)
            if slice_idx is not None:
                crop_name = '{}_{}_{}_{}_{:02d}_{:02d}_{:02d}.tif'.format(cell_type, mode, train_set,
                                                                          frame.split('.tif')[0], slice_idx, y, x)
            else:
                crop_name = '{}_{}_{}_{}_{:02d}_{:02d}.tif'.format(cell_type, mode, train_set,
                                                                   frame.split('.tif')[0], y, x)

            # Check cell number TRA/SEG
            mask_ids = get_nucleus_ids(mask_crop)
            if np.sum(mask_crop[10:-10, 10:-10, 0] > 0) < min_area:  # only cell parts / no cell
                continue
            if len(mask_ids) == 1:  # neigbor may be cut from crop --> set dist to 0
                neighbor_dist_crop = np.zeros_like(neighbor_dist_crop)
            if np.sum(img_crop == 0) > (0.66 * img_crop.shape[0] * img_crop.shape[1]):  # almost background
                # For GOWT1 cells a lot of 0s are in the image
                if np.min(img_crop[:100, :100, ...]) == 0:
                    if np.sum(gaussian_filter(np.squeeze(img_crop), sigma=1) == 0) > (0.66 * img_crop.shape[0] * img_crop.shape[1]):
                        continue
                else:
                    continue
            if np.max(cell_dist_crop) < 0.8:
                continue

            # Remove only partially visible cells in mask for better comparison with tra_gt
            props_crop, n_part = regionprops(mask_crop), 0
            for cell in props_crop:
                if cell.area <= 0.1 * min_area and scale == 1:  # needed since tra_gt seeds are smaller
                    n_part += 1

            tiff.imsave(str(path_train_sets / "{}_{}".format(cell_type, mode) / 'A' / ('img_' + crop_name)), img_crop)
            tiff.imsave(str(path_train_sets / "{}_{}".format(cell_type, mode) / 'A' / ('mask_' + crop_name)), mask_crop)
            tiff.imsave(str(path_train_sets / "{}_{}".format(cell_type, mode) / 'A' / ('dist_cell_' + crop_name)),
                        cell_dist_crop)
            tiff.imsave(str(path_train_sets / "{}_{}".format(cell_type, mode) / 'A' / ('dist_neighbor_' + crop_name)),
                        neighbor_dist_crop)

            running_crop_idx += 1
            if running_crop_idx > max_crops:
                return running_crop_idx

    return running_crop_idx


def get_crop(x, y, crop_size, *imgs):
    """ Get crop from an image

    :param x: Grid position (x-dim).
        :type x: int
    :param y: Grid position (y-dim).
        :type y: int
    :param crop_size: size of the (square) crop
        :type crop_size: int
    :return: img crop.
    :param imgs: Images to crop.
        :type imgs:
    """

    imgs_crop = []

    for img in imgs:
        img_crop = img[y * crop_size:(y + 1) * crop_size, x * crop_size:(x + 1) * crop_size, :]
        imgs_crop.append(img_crop)

    return imgs_crop


def adjust_dimensions(crop_size, *imgs):
    """ Adjust dimensions so that only 'complete' crops are generated.

    :param crop_size: Size of the (square) crops.
        :type crop_size: int
    :param imgs: Images to adjust the dimensions.
        :type imgs:
    :return: img with adjusted dimension.
    """

    img_adj = []

    # Add pseudo color channels
    for img in imgs:
        img = np.expand_dims(img, axis=-1)

        pads = []
        for i in range(2):
            if img.shape[i] < crop_size:
                pads.append((0, crop_size - (img.shape[i] % crop_size)))
            elif img.shape[i] == crop_size:
                pads.append((0, 0))
            else:
                if (img.shape[i] % crop_size) < 0.075 * img.shape[i]:
                    idx_start = (img.shape[i] % crop_size) // 2
                    idx_end = img.shape[i] - ((img.shape[i] % crop_size) - idx_start)
                    if i == 0:
                        img = img[idx_start:idx_end, ...]
                    else:
                        img = img[:, idx_start:idx_end, ...]
                    pads.append((0, 0))
                else:
                    pads.append((0, crop_size - (img.shape[i] % crop_size)))

        img = np.pad(img, (pads[0], pads[1], (0, 0)), mode='constant')

        img_adj.append(img)

    return img_adj


def downscale(img, seg_gt, scale, tra_gt=None):
    """ Downscale image and segmentation ground truth.

    :param img: Image to downscale
        :type img:
    :param seg_gt: Segmentation ground truth to downscale (may be 2D also when img is 3D!)
        :type seg_gt:
    :param scale: Scale factor <= 1.
        :type scale: float
    :param tra_gt: Tracking ground truth
        :type tra_gt:
    :return: downscale images.
    """

    if len(img.shape) == 3:
        scale_img = (1, scale, scale)
    else:
        scale_img = (scale, scale)

    if len(seg_gt.shape) == 3:
        scale_seg = scale_img
    else:
        scale_seg = (scale, scale)

    img = rescale(img, scale=scale_img, order=2, preserve_range=True).astype(img.dtype)
    seg_gt = rescale(seg_gt, scale=scale_seg, order=0, preserve_range=True, anti_aliasing=False).astype(seg_gt.dtype)

    if tra_gt is not None:
        tra_gt = rescale(tra_gt, scale=scale_img, order=0, preserve_range=True, anti_aliasing=False).astype(tra_gt.dtype)

    return img, seg_gt, tra_gt


def make_train_dirs(path, cell_type, mode):
    """ Make directories to save the created training data into.

    :param path: Path to the created training date sets.
        :type path: pathlib Path object.
    :param cell_type: Cell type.
        :type cell_type: str
    :param mode: Primary Track mode ('GT', 'ST', 'GT+ST', 'allGT', 'allST', 'allGT+allST') needed for filename.
        :type mode: str
    :return: None
    """

    if "all" in mode:
        Path.mkdir(path / "{}".format(mode), exist_ok=True, parents=True)
        Path.mkdir(path / "{}".format(mode) / 'train', exist_ok=True)
        Path.mkdir(path / "{}".format(mode) / 'val', exist_ok=True)
    else:
        Path.mkdir(path / "{}_{}".format(cell_type, mode), exist_ok=True, parents=True)
        Path.mkdir(path / "{}_{}".format(cell_type, mode) / 'train', exist_ok=True, parents=True)
        Path.mkdir(path / "{}_{}".format(cell_type, mode) / 'val', exist_ok=True, parents=True)
        Path.mkdir(path / "{}_{}".format(cell_type, mode) / 'A', exist_ok=True, parents=True)
    if mode == 'GT':
        Path.mkdir(path / "{}_{}".format(cell_type, mode) / 'A', exist_ok=True)
        Path.mkdir(path / "{}_{}".format(cell_type, mode) / 'B', exist_ok=True)
        Path.mkdir(path / "{}_{}".format(cell_type, mode) / 'C', exist_ok=True)
        Path.mkdir(path / "{}_{}".format(cell_type, mode) / 'D', exist_ok=True)
        Path.mkdir(path / "{}_{}".format(cell_type, mode) / 'train', exist_ok=True)
        Path.mkdir(path / "{}_{}".format(cell_type, mode) / 'val', exist_ok=True)

    return None


def get_train_val_split(cell_type, mode):
    """ Load training/validation split ids.

    :param cell_type: Cell type.
        :type cell_type: str
    :param mode: Primary Track mode ('GT', 'ST', 'GT+ST', 'allGT', 'allST', 'allGT+allST') needed for filename.
        :type mode: str
    :return: training/validation ids (dict).
    """

    if 'all' in mode:
        if (Path.cwd() / 'segmentation' / 'training' / 'splits' / 'ids_{}.json'.format(mode)).exists():
            with open(Path.cwd() / 'segmentation' / 'training' / 'splits' / 'ids_{}.json'.format(mode)) as f:
                train_val_ids = json.load(f)
        else:
            train_val_ids = {}
    else:
        # If train / val split is available: use it, if not: random
        if (Path.cwd() / 'segmentation' / 'training' / 'splits' / 'ids_{}_{}.json'.format(cell_type, mode)).exists():
            with open(Path.cwd() / 'segmentation' / 'training' / 'splits' / 'ids_{}_{}.json'.format(cell_type, mode)) as f:
                train_val_ids = json.load(f)
        else:
            train_val_ids = {}

    return train_val_ids


def get_gt_settings(gt_id_list):
    """ Calculate parameters needed for training data creation.

    :param gt_id_list: List of all segmentation GT ids.
        :type gt_id_list: list
    :return: search radius, minimum area, max major axis length, scale factor
    """

    # Load all GT and get cell sizes for radius parameter of label creation
    diameters, major_axes, areas = [], [], []

    for gt_id in gt_id_list:
        seg_gt = tiff.imread(str(gt_id))

        if len(seg_gt.shape) == 3:
            for i in range(len(seg_gt)):
                props = regionprops(seg_gt[i])
                for cell in props:  # works not as intended for 3D GTs
                    major_axes.append(cell.major_axis_length)
                    diameters.append(cell.equivalent_diameter)
                    areas.append(cell.area)
        else:
            props = regionprops(seg_gt)
            for cell in props:
                major_axes.append(cell.major_axis_length)
                diameters.append(cell.equivalent_diameter)
                areas.append(cell.area)

    max_diameter, min_diameter = int(np.ceil(np.max(np.array(diameters)))), int(np.ceil(np.min(np.array(diameters))))
    mean_diameter, std_diameter = int(np.ceil(np.mean(np.array(diameters)))), int(np.std(np.array(diameters)))
    max_mal = int(np.ceil(np.max(np.array(major_axes))))
    min_area = int(0.95 * np.floor(np.min(np.array(areas))))

    search_radius = mean_diameter + std_diameter

    if max_diameter > 200 and min_diameter > 35:  # Some simple heuristics for predictions of large cells
        if max_mal > 2 * max_diameter:  # very longish and long cells not made for neighbor distance
            scale = 0.5
            search_radius = min_diameter + 0.5 * std_diameter
        elif max_diameter > 300 and min_diameter > 60:
            scale = 0.5
        elif max_diameter > 250 and min_diameter > 50:
            scale = 0.6
        else:
            scale = 0.7
        min_area = (scale ** 2) * min_area
        max_mal = int(np.ceil(scale * max_mal))
        search_radius = int(np.ceil(scale * search_radius))

    else:
        scale = 1

    return search_radius, min_area, max_mal, scale


def foi_correction_train_st(cell_type, *imgs):
    """ Field of interest correction (see
    https://public.celltrackingchallenge.net/documents/Naming%20and%20file%20content%20conventions.pdf and
    https://public.celltrackingchallenge.net/documents/Annotation%20procedure.pdf )

    (Differs to foi_correction_train for some cell types since some GT training data sets were already fixed before
    we have seen the need for the foi correction. However, should not affect the results)

    :param cell_type: Cell type.
        :type cell_type: str
    :param imgs: Image/mask/ST.
        :type imgs:
    :return: foi corrected image.
    """

    if cell_type in ['Fluo-C2DL-Huh7', 'Fluo-N2DH-GOWT1', 'Fluo-N3DH-CHO', 'PhC-C2DH-U373', 'Fluo-C3DH-H157',
                     'Fluo-N3DH-CHO']:
        E = 50
    elif cell_type in ['Fluo-N2DL-HeLa', 'PhC-C2DL-PSC', 'Fluo-C3DL-MDA231']:
        E = 25
    else:
        E = 0

    img_corr = []

    for img in imgs:
        if len(img.shape) == 2:
            img_corr.append(img[E:img.shape[0] - E, E:img.shape[1] - E])
        else:
            img_corr.append(img[:, E:img.shape[1] - E, E:img.shape[2] - E])

    return img_corr


def foi_correction_train(cell_type, *imgs):
    """ Field of interest correction (see
    https://public.celltrackingchallenge.net/documents/Naming%20and%20file%20content%20conventions.pdf and
    https://public.celltrackingchallenge.net/documents/Annotation%20procedure.pdf )

    :param cell_type: Cell type.
        :type cell_type: str
    :param imgs: Image/mask/GT.
        :type imgs:
    :return: foi corrected image.
    """

    if cell_type in ['Fluo-C2DL-Huh7', 'Fluo-N2DH-GOWT1', 'Fluo-N3DH-CHO', 'PhC-C2DH-U373']:
        E = 50
    elif cell_type in ['Fluo-N2DL-HeLa', 'PhC-C2DL-PSC', 'Fluo-C3DL-MDA231']:
        E = 25
    else:
        E = 0

    img_corr = []

    for img in imgs:
        if len(img.shape) == 2:
            img_corr.append(img[E:img.shape[0] - E, E:img.shape[1] - E])
        else:
            img_corr.append(img[:, E:img.shape[1] - E, E:img.shape[2] - E])

    return img_corr


def get_train_val_frames(train_val_ids):
    """ Get frames used in training/validation split.

    :param train_val_ids: Training/validation split ids.
        :type train_val_ids: dict
    :return: frames used.
    """

    train_val_frames = []
    for split_mode in ['train', 'val']:
        for idx in train_val_ids[split_mode]:
            if '2D' in idx:
                train_val_frames.append('{}_{}'.format(idx.split('_')[-4], idx.split('_')[-3]))
            else:
                train_val_frames.append('{}_{}'.format(idx.split('_')[-5], idx.split('_')[-4]))

    train_val_frames = set(train_val_frames)

    return train_val_frames


def create_ctc_training_sets(path_data, path_train_sets, cell_types):
    """ Create training sets for the Cell Tracking Challenge.

    (needs some revision sometime to make the code readable ...)

    :param path_data: Path to the directory containing the Cell Tracking Challenge training sets.
        :type path_data: Path
    :param path_train_sets: Path to save the training sets into.
        :type path_train_sets: Path
    :return: None
    """

    crop_size = 320

    modes = ['GT', 'ST', 'GT+ST', 'allGT', 'allST', 'allGT+allST']

    for cell_type in cell_types:

        for mode in modes:

            # Check if data set already exists
            if len(list((path_train_sets / "{}_{}".format(cell_type, mode) / 'train').glob('*.tif'))) > 0:
                continue

            if "all" not in mode:
                print('   ... {}: {} set ...'.format(cell_type, mode))

            make_train_dirs(path=path_train_sets, cell_type=cell_type, mode=mode)

            if mode == 'GT':  # Check with TRA-GT if image is fully annotated

                # If train / val split is available: use it, if not: random
                train_val_ids = get_train_val_split(cell_type=cell_type, mode=mode)

                # Get ids of segmentation ground truth
                seg_gt_ids_01 = sorted((path_data / 'training_datasets' / cell_type / '01_GT' / 'SEG').glob('*.tif'))
                seg_gt_ids_02 = sorted((path_data / 'training_datasets' / cell_type / '02_GT' / 'SEG').glob('*.tif'))
                seg_gt_ids = seg_gt_ids_01 + seg_gt_ids_02

                # Get some settings for train data generation
                search_radius, min_area, max_mal, scale = get_gt_settings(gt_id_list=seg_gt_ids)

                # go through files and load SEG and TRA GT
                for seg_gt_id in seg_gt_ids:
                    train_set = seg_gt_id.parents[1].stem.split('_')[0]
                    if len(seg_gt_id.stem.split('_')) > 2:  # only slice annotated
                        frame = seg_gt_id.stem.split('_')[2] + '.tif'
                        slice_idx = int(seg_gt_id.stem.split('_')[3])
                    else:
                        frame = seg_gt_id.name.split('man_seg')[-1]
                    seg_gt = tiff.imread(str(seg_gt_id))
                    img = tiff.imread(str(seg_gt_id.parents[2] / train_set / "t{}".format(frame)))
                    tra_gt = tiff.imread(str(seg_gt_id.parents[1] / 'TRA' / "man_track{}".format(frame)))

                    img, seg_gt, tra_gt = foi_correction_train(cell_type, img, seg_gt, tra_gt)

                    if scale != 1:
                        img, seg_gt, tra_gt = downscale(img=img, seg_gt=seg_gt, scale=scale, tra_gt=tra_gt)

                    # min-max normalize image to 0 - 65535
                    img = 65535 * (img.astype(np.float32) - img.min()) / (img.max() - img.min())
                    img = np.clip(img, 0, 65535).astype(np.uint16)

                    if len(seg_gt.shape) == 3:
                        for i in range(len(seg_gt)):
                            img_slice = img[i].copy()
                            mask = seg_gt[i].copy()
                            if np.max(mask) == 0:  # empty frame
                                continue
                            nucleus_ids = get_nucleus_ids(mask)
                            hlabel = np.zeros(shape=mask.shape, dtype=mask.dtype)
                            for nucleus_id in nucleus_ids:
                                hlabel += nucleus_id * binary_closing(mask == nucleus_id, np.ones((5, 5))).astype(
                                    mask.dtype)
                            mask = hlabel
                            tr_gt_slice = mask.copy()  # assumption: in 3D GT annotations all cells are annotated
                            generate_data(img=img_slice, mask=mask, tra_gt=tr_gt_slice, search_radius=search_radius,
                                          max_mal=max_mal, crop_size=crop_size, cell_type=cell_type, mode=mode,
                                          train_set=train_set, frame=frame, min_area=min_area, scale=scale,
                                          path_train_sets=path_train_sets, slice_idx=i)
                    else:
                        if '3D' in cell_type:
                            img = img[slice_idx]
                            # Needed seed could be outside the slice --> maximum intensity projection
                            slice_min = np.maximum(slice_idx-2, 0)
                            slice_max = np.minimum(slice_idx+2, len(img)-1)
                            tra_gt = np.max(tra_gt[slice_min:slice_max], axis=0)  # best bring seed size to min_area ...
                            nucleus_ids = get_nucleus_ids(seg_gt)
                            hlabel = np.zeros(shape=seg_gt.shape, dtype=seg_gt.dtype)
                            for nucleus_id in nucleus_ids:
                                hlabel += nucleus_id * binary_closing(seg_gt == nucleus_id, np.ones((5, 5))).astype(
                                    seg_gt.dtype)
                            seg_gt = hlabel

                        generate_data(img=img, mask=seg_gt, tra_gt=tra_gt, search_radius=search_radius,
                                      max_mal=max_mal, crop_size=crop_size, cell_type=cell_type, mode=mode,
                                      train_set=train_set, frame=frame, min_area=min_area, scale=scale,
                                      path_train_sets=path_train_sets)

                train_data_info = {'scale': scale,
                                   'max_mal': max_mal,
                                   'min_area': min_area,
                                   'search_radius': search_radius}
                with open(path_train_sets / "{}_{}".format(cell_type, mode) / 'info.json', 'w', encoding='utf-8') as outfile:
                    json.dump(train_data_info, outfile, ensure_ascii=False, indent=2)

                # train/val splits
                img_ids = sorted((path_train_sets / "{}_{}".format(cell_type, mode) / 'A').glob('img*.tif'))
                if len(img_ids) <= 30:  # Use also "B" quality images when too few "A" quality images are available
                    img_ids_B = sorted((path_train_sets / "{}_{}".format(cell_type, mode) / 'B').glob('img*.tif'))
                else:
                    img_ids_B = []

                if not train_val_ids:  # no split available

                    img_ids_stem = []
                    for idx in img_ids:
                        img_ids_stem.append(idx.stem.split('img_')[-1])

                    # Random 80%/20% split
                    shuffle(img_ids_stem)
                    train_ids = img_ids_stem[0:int(np.floor(0.8 * len(img_ids_stem)))]
                    val_ids = img_ids_stem[int(np.floor(0.8 * len(img_ids_stem))):]

                    # Add "B" quality only to train
                    for idx in img_ids_B:
                        train_ids.append(idx.stem.split('img_')[-1])

                    train_val_ids = {'train': train_ids, 'val': val_ids}
                    with open(Path.cwd() / "segmentation" / "training" / "splits" / 'ids_{}_{}.json'.format(cell_type, mode),
                              'w', encoding='utf-8') as outfile:
                        json.dump(train_val_ids, outfile, ensure_ascii=False, indent=2)

                for train_mode in ['train', 'val']:
                    for idx in train_val_ids[train_mode]:
                        source_path = path_train_sets / "{}_{}".format(cell_type, mode)
                        target_path = path_train_sets / "{}_{}".format(cell_type, mode) / train_mode
                        if (source_path / "A" / ("img_{}.tif".format(idx))).exists():
                            source_path = source_path / "A"
                        else:
                            source_path = source_path / "B"
                        shutil.copyfile(str(source_path / "img_{}.tif".format(idx)),
                                        str(target_path / "img_{}.tif".format(idx)))
                        shutil.copyfile(str(source_path / "dist_cell_{}.tif".format(idx)),
                                        str(target_path / "dist_cell_{}.tif".format(idx)))
                        shutil.copyfile(str(source_path / "dist_neighbor_{}.tif".format(idx)),
                                        str(target_path / "dist_neighbor_{}.tif".format(idx)))
                        shutil.copyfile(str(source_path / "mask_{}.tif".format(idx)),
                                        str(target_path / "mask_{}.tif".format(idx)))

            if mode == 'ST':  # Check with TRA-GT if image is fully annotated

                # Maximum number of crops
                n_max = 280

                # If train / val split is available: use it, if not: random
                train_val_ids = get_train_val_split(cell_type=cell_type, mode=mode)

                # Get ids of segmentation ground truth
                seg_st_ids_01 = sorted((path_data / 'training_datasets' / cell_type / '01_ST' / 'SEG').glob('*.tif'))
                seg_st_ids_02 = sorted((path_data / 'training_datasets' / cell_type / '02_ST' / 'SEG').glob('*.tif'))
                seg_st_ids = seg_st_ids_01 + seg_st_ids_02

                # Get some settings for train data generation
                search_radius, min_area, max_mal, scale = get_gt_settings(gt_id_list=seg_st_ids)

                if not train_val_ids:
                    if len(seg_st_ids) > n_max // 2:
                        if '3D' in cell_type:  # 3D images are in the end maybe not that good segmented:
                            seg_st_ids = seg_st_ids_01[:int(n_max//2.5)] + seg_st_ids_02[:int(n_max//2.5)]
                        else:
                            if len(seg_st_ids) > 1000:
                                seg_st_ids = seg_st_ids_01[:1000:10] + seg_st_ids_01[1000::5] + \
                                             seg_st_ids_02[:1000:10] + seg_st_ids_02[1000::5]
                            else:
                                seg_st_ids = seg_st_ids[::2]

                    if '3D' in cell_type:
                        if len(tiff.imread(str(seg_st_ids[0]))) > 40:  # reduce computation time
                            seg_st_ids = seg_st_ids[::2]
                            slice_increment = 4
                        elif len(tiff.imread(str(seg_st_ids[0]))) > 30:  # reduce computation time
                            seg_st_ids = seg_st_ids[::2]
                            slice_increment = 2
                        else:
                            slice_increment = 1
                else:
                    train_val_frames = get_train_val_frames(train_val_ids)
                    if '3D' in cell_type:
                        if len(tiff.imread(str(seg_st_ids[0]))) > 40:  # reduce computation time
                            slice_increment = 4
                        elif len(tiff.imread(str(seg_st_ids[0]))) > 30:  # reduce computation time
                            slice_increment = 2
                        else:
                            slice_increment = 1

                if '2D' in cell_type:
                    if not train_val_ids:
                        # Shuffle ids
                        shuffle(seg_st_ids)
                    else:
                        n_max = 1e6  # create all crops to make sure that each needed crop is available

                # go through files and load SEG and TRA GT
                running_index = 0  # count crops
                for seg_st_id in seg_st_ids:

                    if running_index > n_max:
                        continue

                    if train_val_ids:
                        if '{}_{}'.format(seg_st_id.parents[1].stem.split('_')[0], seg_st_id.stem.split('man_seg')[-1]) in train_val_frames:
                            pass
                        else:
                            continue

                    train_set = seg_st_id.parents[1].stem.split('_')[0]
                    frame = seg_st_id.name.split('man_seg')[-1]
                    seg_st = tiff.imread(str(seg_st_id))
                    img = tiff.imread(str(seg_st_id.parents[2] / train_set / "t{}".format(frame)))

                    img, seg_st = foi_correction_train_st(cell_type, img, seg_st)

                    if scale != 1:
                        img, seg_gt, _ = downscale(img=img, seg_gt=seg_st, scale=scale)

                    # min-max normalize image to 0 - 65535
                    img = 65535 * (img.astype(np.float32) - img.min()) / (img.max() - img.min())
                    img = np.clip(img, 0, 65535).astype(np.uint16)

                    if len(seg_st.shape) == 3:
                        # Select two random slices which contain cells
                        img_mean, img_std = np.mean(img), np.std(img)
                        for i in range(len(img)):  # create data for all images ...
                            if i % slice_increment == 0:
                                if slice_increment > 1:
                                    if np.mean(img[i]) < img_mean + 0.1 * img_std or np.sum(seg_st[i] == 0) < 0.02 * img.shape[1] * img.shape[2]:
                                        continue
                                else:
                                    if np.mean(img[i]) < img_mean - 0.1 * img_std or np.sum(seg_st[i] > 0) < 0.02 * img.shape[1] * img.shape[2]:
                                        continue
                                # Get slices
                                img_slice = img[i]
                                seg_st_slice = seg_st[i]

                                # Opening + closing
                                if cell_type in ['Fluo-C3DH-H157', 'Fluo-N3DH-CHO']:
                                    kernel_closing = np.ones((20, 20))
                                    kernel_opening = np.ones((20, 20))
                                elif cell_type == 'Fluo-C3DL-MDA231':
                                    kernel_closing = np.ones((3, 3))
                                    kernel_opening = np.ones((3, 3))
                                elif cell_type == 'Fluo-N3DH-CE':
                                    kernel_closing = np.ones((15, 15))
                                    kernel_opening = np.ones((15, 15))
                                else:
                                    kernel_closing = np.ones((10, 10))
                                    kernel_opening = np.ones((10, 10))
                                seg_st_slice = close_st(seg_st_slice, kernel_closing, kernel_opening)

                                if cell_type == 'Fluo-N3DH-CE':
                                    props = regionprops(seg_st_slice)
                                    for nucleus in props:
                                        if nucleus.bbox_area < 20 * 20:
                                            seg_st_slice[seg_st_slice == nucleus.label] = 0

                                # No running index --> create all crops and select later
                                _ = generate_data_st(img=img_slice, mask=seg_st_slice, search_radius=search_radius,
                                                     max_mal=max_mal, crop_size=crop_size, cell_type=cell_type,
                                                     mode=mode, train_set=train_set, frame=frame,  min_area=min_area,
                                                     scale=scale, path_train_sets=path_train_sets, slice_idx=i,
                                                     running_crop_idx=0, max_crops=n_max)
                    else:

                        if cell_type == 'DIC-C2DH-HeLa':
                            seg_st = close_st(seg_st=seg_st)

                        running_index = generate_data_st(img=img, mask=seg_st, search_radius=search_radius,
                                                         max_mal=max_mal, crop_size=crop_size, cell_type=cell_type,
                                                         mode=mode, train_set=train_set, frame=frame, min_area=min_area,
                                                         scale=scale, path_train_sets=path_train_sets,
                                                         running_crop_idx=running_index, max_crops=n_max)

                train_data_info = {'scale': scale,
                                   'max_mal': max_mal,
                                   'min_area': min_area,
                                   'search_radius': search_radius}
                with open(path_train_sets / "{}_{}".format(cell_type, mode) / 'info.json', 'w', encoding='utf-8') as outfile:
                    json.dump(train_data_info, outfile, ensure_ascii=False, indent=2)

                # train/val splits
                img_ids = sorted((path_train_sets / "{}_{}".format(cell_type, mode) / 'A').glob('img*.tif'))

                if not train_val_ids:  # no split available

                    img_ids_stem = []
                    for idx in img_ids:
                        img_ids_stem.append(idx.stem.split('img_')[-1])

                    if '3D' in cell_type:
                        shuffle(img_ids_stem)
                        img_ids_stem = img_ids_stem[:n_max]

                    # Random 80%/20% split
                    shuffle(img_ids_stem)
                    train_ids = img_ids_stem[0:int(np.floor(0.8 * len(img_ids_stem)))]
                    val_ids = img_ids_stem[int(np.floor(0.8 * len(img_ids_stem))):]

                    train_val_ids = {'train': train_ids, 'val': val_ids}
                    with open(Path.cwd() / "segmentation" / "training" / "splits" / 'ids_{}_{}.json'.format(cell_type, mode),
                              'w', encoding='utf-8') as outfile:
                        json.dump(train_val_ids, outfile, ensure_ascii=False, indent=2)

                for train_mode in ['train', 'val']:
                    for idx in train_val_ids[train_mode]:
                        source_path = path_train_sets / "{}_{}".format(cell_type, mode)
                        target_path = path_train_sets / "{}_{}".format(cell_type, mode) / train_mode
                        source_path = source_path / "A"

                        shutil.copyfile(str(source_path / "img_{}.tif".format(idx)),
                                        str(target_path / "img_{}.tif".format(idx)))
                        shutil.copyfile(str(source_path / "dist_cell_{}.tif".format(idx)),
                                        str(target_path / "dist_cell_{}.tif".format(idx)))
                        shutil.copyfile(str(source_path / "dist_neighbor_{}.tif".format(idx)),
                                        str(target_path / "dist_neighbor_{}.tif".format(idx)))
                        shutil.copyfile(str(source_path / "mask_{}.tif".format(idx)),
                                        str(target_path / "mask_{}.tif".format(idx)))

            elif mode == 'GT+ST':  # Basis is the corresponding GT set

                # If train / val split is available: use it, if not: random
                train_val_ids = get_train_val_split(cell_type=cell_type, mode=mode)

                if not train_val_ids:
                    random_split = True
                else:
                    random_split = False

                # Copy GT data set
                shutil.rmtree(str(path_train_sets / "{}_{}".format(cell_type, mode) / 'train'))
                shutil.rmtree(str(path_train_sets / "{}_{}".format(cell_type, mode) / 'val'))
                shutil.copytree(str(path_train_sets / "{}_GT".format(cell_type) / 'train'),
                                str(path_train_sets / "{}_{}".format(cell_type, mode) / 'train'), )
                shutil.copytree(str(path_train_sets / "{}_GT".format(cell_type) / 'val'),
                                str(path_train_sets / "{}_{}".format(cell_type, mode) / 'val'))

                # Get number of GT crops (train/val)
                num_gt_train = len(sorted((path_train_sets / "{}_{}".format(cell_type, mode) / 'train').glob('img*.tif')))
                num_gt_val = len(sorted((path_train_sets / "{}_{}".format(cell_type, mode) / 'val').glob('img*.tif')))
                num_add_st_train = np.maximum(int(0.33 * num_gt_train), 75 - num_gt_train)
                num_add_st_val = np.maximum(int(0.25 * num_gt_val), 15 - num_gt_val)
                if cell_type in ['Fluo-C3DH-H157', 'Fluo-C2DL-MSC']:  # just use all ST due to different scaling
                    num_add_st_train = 1e3
                    num_add_st_val = 1e3

                # Get existing GT frames
                gt_frames_used = get_gt_frames(path_data / 'training_datasets' / cell_type)

                # Temporary copy the ST train and val set
                shutil.copytree(str(path_train_sets / "{}_ST".format(cell_type) / 'train'),
                                str(path_train_sets / "{}_{}".format(cell_type, mode) / 'train_ST'))
                shutil.copytree(str(path_train_sets / "{}_ST".format(cell_type) / 'val'),
                                str(path_train_sets / "{}_{}".format(cell_type, mode) / 'val_ST'))

                # Go through ST crops and remove crop if frame exists not as GT
                st_train_ids = sorted((path_train_sets / "{}_{}".format(cell_type, mode) / 'train_ST').glob('img*.tif'))
                st_val_ids = sorted((path_train_sets / "{}_{}".format(cell_type, mode) / 'val_ST').glob('img*.tif'))
                for st_id in (st_train_ids + st_val_ids):
                    frame = str(st_id).split('ST_')[-1].split('_')[0] + '_' + str(st_id).split('ST_')[-1].split('_')[1]
                    if frame in gt_frames_used:
                        files_to_remove = list(st_id.parent.glob("*{}".format(st_id.name.split('img')[-1])))
                        for idx in files_to_remove:
                            os.remove(idx)

                # Get number of usable crops in ST_train/ST_val
                st_train_ids = sorted((path_train_sets / "{}_{}".format(cell_type, mode) / 'train_ST').glob('img*.tif'))
                st_val_ids = sorted((path_train_sets / "{}_{}".format(cell_type, mode) / 'val_ST').glob('img*.tif'))
                shuffle(st_train_ids)
                shuffle(st_val_ids)

                if not train_val_ids:
                    counter, counter_val = 0, 0
                    train_val_ids = {'train_st': [], 'val_st': []}
                else:
                    counter, counter_val = [], []
                for st_train_id in st_train_ids:
                    if isinstance(counter, list):
                        if st_train_id.stem.split('img_')[-1] in train_val_ids['train_st']:
                            pass
                        else:
                            continue
                    else:
                        if counter < num_add_st_train:
                            counter += 1
                        else:
                            continue
                    # Copy img, distance labels and mask
                    train_val_ids['train_st'].append(st_train_id.stem.split('img_')[-1])
                    files_to_copy = list(st_train_id.parent.glob("*{}".format(st_train_id.name.split('img')[-1])))
                    for idx in files_to_copy:
                        shutil.copyfile(str(idx),
                                        str(path_train_sets / "{}_{}".format(cell_type, mode) / 'train' / idx.name))

                for st_val_id in st_val_ids:
                    if isinstance(counter_val, list):
                        if st_val_id.stem.split('img_')[-1] in train_val_ids['val_st']:
                            pass
                        else:
                            continue
                    else:
                        if counter_val < num_add_st_val:
                            counter_val += 1
                        else:
                            continue
                    # Copy img, distance labels and mask
                    train_val_ids['val_st'].append(st_val_id.stem.split('img_')[-1])
                    files_to_copy = list(st_val_id.parent.glob("*{}".format(st_val_id.name.split('img')[-1])))
                    for idx in files_to_copy:
                        shutil.copyfile(str(idx),
                                        str(path_train_sets / "{}_{}".format(cell_type, mode) / 'val' / idx.name))

                train_data_info = {'scale': 1}  # For simplicity just use scale 1 for all cell types
                with open(path_train_sets / "{}_{}".format(cell_type, mode) / 'info.json', 'w',
                          encoding='utf-8') as outfile:
                    json.dump(train_data_info, outfile, ensure_ascii=False, indent=2)

                if random_split:
                    with open(Path.cwd() / "segmentation" / "training" / "splits" / 'ids_{}_{}.json'.format(cell_type, mode),
                              'w', encoding='utf-8') as outfile:
                        json.dump(train_val_ids, outfile, ensure_ascii=False, indent=2)

                # Remove temporary directories
                shutil.rmtree(str(path_train_sets / "{}_{}".format(cell_type, mode) / 'train_ST'))
                shutil.rmtree(str(path_train_sets / "{}_{}".format(cell_type, mode) / 'val_ST'))
                shutil.rmtree(str(path_train_sets / "{}_{}".format(cell_type, mode) / 'A'))

    for mode in modes:

        if mode == 'allGT':  # Copy train and val sets together but avoid to high imbalance of cell types

            cell_counts = {'train': {}, 'val': {}}

            # Check if data set already exists
            if len(list((path_train_sets / "{}".format(mode) / 'train').glob('*.tif'))) > 0:
                continue

            print('   ... allGT set ...')

            train_val_ids = get_train_val_split(cell_type='all', mode=mode)

            if not train_val_ids:
                train_ids, val_ids = {'train': {}, 'val': {}}, {'train:': {}, 'val': {}}

            for cell_type in cell_types:

                if cell_type == 'Fluo-C2DL-MSC':  # use no c2dl-msc data since too different to other cells
                    cell_counts['train'][cell_type], cell_counts['val'][cell_type] = 0, 0
                    continue

                img_ids = {'train': sorted((path_train_sets / "{}_GT".format(cell_type) / 'train').glob('img*.tif')),
                           'val': sorted((path_train_sets / "{}_GT".format(cell_type) / 'val').glob('img*.tif'))}

                if not train_val_ids:
                    if "3D" in cell_type:
                        if len(img_ids['train']) + len(img_ids['val']) > 100:
                            p = 0.15  # use only 20% of images with dist_neighbor = 0
                        elif len(img_ids['train']) + len(img_ids['val']) > 50:
                            p = 0.75  # use 2/3 of images with dist_neighbor = 0
                        else:
                            p = 1
                    else:
                        if len(img_ids['train']) + len(img_ids['val']) > 150:
                            p = 0.5
                        elif len(img_ids['train']) + len(img_ids['val']) > 75:
                            p = 0.75
                        else:
                            p = 1

                    for train_mode in ['train', 'val']:
                        counts = 0
                        hlist = []
                        for idx in img_ids[train_mode]:
                            fname = idx.name.split('img_')[-1]
                            if np.sum(tiff.imread(str(idx.parent / 'dist_neighbor_{}'.format(fname))) > 0) == 0:
                                if random() > p:
                                    continue
                            hlist.append(fname.split('.tif')[0])
                            source_path = path_train_sets / "{}_GT".format(cell_type) / train_mode
                            target_path = path_train_sets / "{}".format(mode) / train_mode
                            shutil.copyfile(str(source_path / "img_{}".format(fname)),
                                            str(target_path / "img_{}".format(fname)))
                            shutil.copyfile(str(source_path / "dist_cell_{}".format(fname)),
                                            str(target_path / "dist_cell_{}".format(fname)))
                            shutil.copyfile(str(source_path / "dist_neighbor_{}".format(fname)),
                                            str(target_path / "dist_neighbor_{}".format(fname)))
                            shutil.copyfile(str(source_path / "mask_{}".format(fname)),
                                            str(target_path / "mask_{}".format(fname)))
                            counts += 1
                        cell_counts[train_mode][cell_type] = counts
                        train_ids[train_mode][cell_type] = deepcopy(hlist)
                else:
                    for train_mode in ['train', 'val']:
                        counts = 0
                        for idx in img_ids[train_mode]:
                            fname = idx.name.split('img_')[-1]
                            if fname.split('.tif')[0] in train_val_ids[train_mode][cell_type]:
                                source_path = path_train_sets / "{}_GT".format(cell_type) / train_mode
                                target_path = path_train_sets / "{}".format(mode) / train_mode
                                shutil.copyfile(str(source_path / "img_{}".format(fname)),
                                                str(target_path / "img_{}".format(fname)))
                                shutil.copyfile(str(source_path / "dist_cell_{}".format(fname)),
                                                str(target_path / "dist_cell_{}".format(fname)))
                                shutil.copyfile(str(source_path / "dist_neighbor_{}".format(fname)),
                                                str(target_path / "dist_neighbor_{}".format(fname)))
                                shutil.copyfile(str(source_path / "mask_{}".format(fname)),
                                                str(target_path / "mask_{}".format(fname)))
                                counts += 1
                        cell_counts[train_mode][cell_type] = counts
            with open(path_train_sets / "{}".format(mode) / 'info.json', 'w', encoding='utf-8') as outfile:
                json.dump(cell_counts, outfile, ensure_ascii=False, indent=2)
            if not train_val_ids:
                with open(Path.cwd() / "segmentation" / "training" / "splits" / 'ids_{}.json'.format(mode),
                          'w', encoding='utf-8') as outfile:
                    json.dump(train_ids, outfile, ensure_ascii=False, indent=2)

        elif mode == 'allST':

            cell_counts = {'train': {}, 'val': {}}

            # Check if data set already exists
            if len(list((path_train_sets / "{}".format(mode) / 'train').glob('*.tif'))) > 0:
                continue

            print('   ... allST set ...')

            train_val_ids = get_train_val_split(cell_type='all', mode=mode)

            if not train_val_ids:
                train_ids, val_ids = {'train': {}, 'val': {}}, {'train:': {}, 'val': {}}

            for cell_type in cell_types:

                img_ids = {
                    'train': sorted((path_train_sets / "{}_ST".format(cell_type) / 'train').glob('img*.tif')),
                    'val': sorted((path_train_sets / "{}_ST".format(cell_type) / 'val').glob('img*.tif'))}

                if not train_val_ids:

                    p_neighbor = 0.9  # keep (almost) all crops with neighbor distance information
                    p_no_neighbor = 0.6  # keep more than half of the other crops

                    if cell_type in ['Fluo-C2DL-MSC', 'DIC-C2DH-HeLa']:
                        p_neighbor = 0.4
                        p_no_neighbor = 0.2

                    if cell_type in ['Fluo-C3DL-MDA231', 'Fluo-C3DH-H157']:
                        p_neighbor = 0.6
                        p_no_neighbor = 0.4

                    for train_mode in ['train', 'val']:
                        counts = 0
                        hlist = []
                        for idx in img_ids[train_mode]:
                            fname = idx.name.split('img_')[-1]
                            if np.sum(tiff.imread(str(idx.parent / 'dist_neighbor_{}'.format(fname))) > 0.5) == 0:
                                if random() > p_no_neighbor:
                                    continue
                            else:
                                if random() > p_neighbor:
                                    continue
                            hlist.append(fname.split('.tif')[0])
                            source_path = path_train_sets / "{}_ST".format(cell_type) / train_mode
                            target_path = path_train_sets / "{}".format(mode) / train_mode
                            shutil.copyfile(str(source_path / "img_{}".format(fname)),
                                            str(target_path / "img_{}".format(fname)))
                            shutil.copyfile(str(source_path / "dist_cell_{}".format(fname)),
                                            str(target_path / "dist_cell_{}".format(fname)))
                            shutil.copyfile(str(source_path / "dist_neighbor_{}".format(fname)),
                                            str(target_path / "dist_neighbor_{}".format(fname)))
                            shutil.copyfile(str(source_path / "mask_{}".format(fname)),
                                            str(target_path / "mask_{}".format(fname)))
                            counts += 1
                        cell_counts[train_mode][cell_type] = counts
                        train_ids[train_mode][cell_type] = deepcopy(hlist)
                else:
                    for train_mode in ['train', 'val']:
                        counts = 0
                        for idx in img_ids[train_mode]:
                            fname = idx.name.split('img_')[-1]
                            if fname.split('.tif')[0] in train_val_ids[train_mode][cell_type]:
                                source_path = path_train_sets / "{}_ST".format(cell_type) / train_mode
                                target_path = path_train_sets / "{}".format(mode) / train_mode
                                shutil.copyfile(str(source_path / "img_{}".format(fname)),
                                                str(target_path / "img_{}".format(fname)))
                                shutil.copyfile(str(source_path / "dist_cell_{}".format(fname)),
                                                str(target_path / "dist_cell_{}".format(fname)))
                                shutil.copyfile(str(source_path / "dist_neighbor_{}".format(fname)),
                                                str(target_path / "dist_neighbor_{}".format(fname)))
                                shutil.copyfile(str(source_path / "mask_{}".format(fname)),
                                                str(target_path / "mask_{}".format(fname)))
                                counts += 1
                        cell_counts[train_mode][cell_type] = counts
            with open(path_train_sets / "{}".format(mode) / 'info.json', 'w', encoding='utf-8') as outfile:
                json.dump(cell_counts, outfile, ensure_ascii=False, indent=2)
            if not train_val_ids:
                with open(Path.cwd() / "segmentation" / "training" / "splits" / 'ids_{}.json'.format(mode),
                          'w', encoding='utf-8') as outfile:
                    json.dump(train_ids, outfile, ensure_ascii=False, indent=2)

        elif mode == 'allGT+allST':

            cell_counts = {'train': {}, 'val': {}}

            # Check if data set already exists
            if len(list((path_train_sets / "{}".format(mode) / 'train').glob('*.tif'))) > 0:
                continue

            print('   ... allGT+allST set ...')

            train_val_ids = get_train_val_split(cell_type='all', mode=mode)

            if not train_val_ids:
                train_val_ids = {'train': [], 'val': []}
                random_split = True
            else:
                random_split = False
                n_max = {'train': 1e5, 'val': 1e5}  # just a high number since only the ids in train_val_ids are used

            for cell_type in cell_types:

                if random_split:
                    n_min_train, n_min_val = 75, 15  # minimum number of images in each GT+ST set
                    n_max = {'train': 2 * n_min_train, 'val': 2 * n_min_val}
                    if '3D' in cell_type or cell_type == 'Fluo-C2DL-MSC':
                        n_max = {'train': n_min_train, 'val': n_min_val}

                img_ids = {'train': sorted((path_train_sets / "{}_GT+ST".format(cell_type) / 'train').glob('img*.tif')),
                           'val': sorted((path_train_sets / "{}_GT+ST".format(cell_type) / 'val').glob('img*.tif'))}

                if cell_type == 'Fluo-C2DL-MSC':
                    img_ids = {'train': sorted((path_train_sets / "{}_GT+ST".format(cell_type) / 'train').glob('img*ST*.tif')),
                               'val': sorted((path_train_sets / "{}_GT+ST".format(cell_type) / 'val').glob('img*ST*.tif'))}

                shuffle(img_ids['train'])
                shuffle(img_ids['val'])

                counter = {'train': 0, 'val': 0}
                for train_mode in ['train', 'val']:

                    for img_id in img_ids[train_mode]:
                        if random_split:
                            if counter[train_mode] < n_max[train_mode]:
                                counter[train_mode] += 1
                            else:
                                continue
                        else:  # ids available
                            if img_id.stem.split('img_')[-1] in train_val_ids[train_mode]:
                                counter[train_mode] += 1
                            else:
                                continue
                        # Copy img, distance labels and mask
                        if random_split:
                            train_val_ids[train_mode].append(img_id.stem.split('img_')[-1])
                        files_to_copy = list(img_id.parent.glob("*{}".format(img_id.name.split('img')[-1])))
                        for idx in files_to_copy:
                            shutil.copyfile(str(idx), str(path_train_sets / "{}".format(mode) / train_mode / idx.name))
                    cell_counts[train_mode][cell_type] = counter[train_mode]

            with open(path_train_sets / "{}".format(mode) / 'info.json', 'w', encoding='utf-8') as outfile:
                json.dump(cell_counts, outfile, ensure_ascii=False, indent=2)
            if random_split:  # ids not available
                with open(Path.cwd() / "segmentation" / "training" / "splits" / 'ids_{}.json'.format(mode),
                          'w', encoding='utf-8') as outfile:
                    json.dump(train_val_ids, outfile, ensure_ascii=False, indent=2)

    return None


def create_sim_training_sets(path_data, path_train_sets):
    """ Create training sets for the simulated Cell Tracking Challenge data sets Fluo-N2DH-SIM+ & Fluo-N3DH-SIM+.

    (needs some revision sometime to make the code readable ...)

    :param path_data: Path to the directory containing the Cell Tracking Challenge training sets.
        :type path_data: Path
    :param path_train_sets: Path to save the training sets into.
        :type path_train_sets: Path
    :return: None
    """

    crop_size = 320

    cell_types = ['Fluo-N2DH-SIM+', 'Fluo-N3DH-SIM+']

    modes = ['GT']

    for cell_type in cell_types:

        for mode in modes:

            # Check if data set already exists
            if len(list((path_train_sets / "{}_{}".format(cell_type, mode) / 'train').glob('*.tif'))) > 0:
                continue

            if "all" not in mode:
                print('   ... {}: {} set ...'.format(cell_type, mode))

            make_train_dirs(path=path_train_sets, cell_type=cell_type, mode=mode)

            if cell_type == 'Fluo-N3DH-SIM+':
                shutil.copytree(str(path_train_sets / 'Fluo-N2DH-SIM+_GT' / 'train'),
                                str(path_train_sets / 'Fluo-N3DH-SIM+_GT' / 'train'),
                                dirs_exist_ok=True)
                shutil.copytree(str(path_train_sets / 'Fluo-N2DH-SIM+_GT' / 'val'),
                                str(path_train_sets / 'Fluo-N3DH-SIM+_GT' / 'val'),
                                dirs_exist_ok=True)

            if mode == 'GT':  # Simulated data sets are always fully annotated

                # If train / val split is available: use it, if not: random
                train_val_ids = get_train_val_split(cell_type=cell_type, mode=mode)

                # Get ids of segmentation ground truth
                seg_gt_ids_01 = sorted((path_data / 'training_datasets' / cell_type / '01_GT' / 'SEG').glob('*.tif'))
                seg_gt_ids_02 = sorted((path_data / 'training_datasets' / cell_type / '02_GT' / 'SEG').glob('*.tif'))
                if '3D' in cell_type:
                    seg_gt_ids_01 = seg_gt_ids_01[::3]
                    seg_gt_ids_02 = seg_gt_ids_02[::2]
                seg_gt_ids = seg_gt_ids_01 + seg_gt_ids_02

                # Get some settings for train data generation
                search_radius, min_area, max_mal, scale = get_gt_settings(gt_id_list=seg_gt_ids)

                # go through files and load SEG and TRA GT
                slice_idx = 0
                for seg_gt_id in seg_gt_ids:
                    train_set = seg_gt_id.parents[1].stem.split('_')[0]
                    if len(seg_gt_id.stem.split('_')) > 2:  # only slice annotated
                        frame = seg_gt_id.stem.split('_')[2] + '.tif'
                        slice_idx = int(seg_gt_id.stem.split('_')[3])
                    else:
                        frame = seg_gt_id.name.split('man_seg')[-1]
                    seg_gt = tiff.imread(str(seg_gt_id))
                    img = tiff.imread(str(seg_gt_id.parents[2] / train_set / "t{}".format(frame)))

                    if scale != 1:
                        img, seg_gt, _ = downscale(img=img, seg_gt=seg_gt, scale=scale)

                    # min-max normalize image to 0 - 65535
                    img = 65535 * (img.astype(np.float32) - img.min()) / (img.max() - img.min())
                    img = np.clip(img, 0, 65535).astype(np.uint16)

                    if len(seg_gt.shape) == 3:
                        for i in range(len(seg_gt)):
                            img_slice = img[i].copy()
                            mask = seg_gt[i].copy()
                            if np.max(mask) == 0:  # empty slice
                                continue
                            else:
                                if slice_idx % 4 == 0:  # do not create for each slice training data
                                    slice_idx += 1
                                    pass
                                else:
                                    slice_idx += 1
                                    continue
                            nucleus_ids = get_nucleus_ids(mask)
                            hlabel = np.zeros(shape=mask.shape, dtype=mask.dtype)
                            for nucleus_id in nucleus_ids:
                                hlabel += nucleus_id * binary_closing(mask == nucleus_id, np.ones((5, 5))).astype(
                                    mask.dtype)
                            mask = hlabel
                            tr_gt_slice = mask.copy()  # assumption: in 3D GT annotations all cells are annotated
                            generate_data(img=img_slice, mask=mask, tra_gt=tr_gt_slice, search_radius=search_radius,
                                          max_mal=max_mal, crop_size=crop_size, cell_type=cell_type, mode=mode,
                                          train_set=train_set, frame=frame, min_area=min_area, scale=scale,
                                          path_train_sets=path_train_sets, slice_idx=i)
                    else:
                        if '3D' in cell_type:
                            img = img[slice_idx]
                            # Needed seed could be outside the slice --> maximum intensity projection
                            slice_min = np.maximum(slice_idx-2, 0)
                            slice_max = np.minimum(slice_idx+2, len(img)-1)
                            tra_gt = np.max(tra_gt[slice_min:slice_max], axis=0)  # best bring seed size to min_area ...
                            nucleus_ids = get_nucleus_ids(seg_gt)
                            hlabel = np.zeros(shape=seg_gt.shape, dtype=seg_gt.dtype)
                            for nucleus_id in nucleus_ids:
                                hlabel += nucleus_id * binary_closing(seg_gt == nucleus_id, np.ones((5, 5))).astype(
                                    seg_gt.dtype)
                            seg_gt = hlabel

                        tra_gt = seg_gt.copy()

                        generate_data(img=img, mask=seg_gt, tra_gt=tra_gt, search_radius=search_radius,
                                      max_mal=max_mal, crop_size=crop_size, cell_type=cell_type, mode=mode,
                                      train_set=train_set, frame=frame, min_area=min_area, scale=scale,
                                      path_train_sets=path_train_sets)

                train_data_info = {'scale': scale,
                                   'max_mal': max_mal,
                                   'min_area': min_area,
                                   'search_radius': search_radius}
                with open(path_train_sets / "{}_{}".format(cell_type, mode) / 'info.json', 'w', encoding='utf-8') as outfile:
                    json.dump(train_data_info, outfile, ensure_ascii=False, indent=2)

                # train/val splits
                img_ids = sorted((path_train_sets / "{}_{}".format(cell_type, mode) / 'A').glob('img*.tif'))
                if len(img_ids) <= 30:  # Use also "B" quality images when too few "A" quality images are available
                    img_ids_B = sorted((path_train_sets / "{}_{}".format(cell_type, mode) / 'B').glob('img*.tif'))
                else:
                    img_ids_B = []

                if not train_val_ids:  # no split available

                    img_ids_stem = []
                    for idx in img_ids:
                        img_ids_stem.append(idx.stem.split('img_')[-1])

                    # Random 80%/20% split
                    shuffle(img_ids_stem)
                    train_ids = img_ids_stem[0:int(np.floor(0.8 * len(img_ids_stem)))]
                    val_ids = img_ids_stem[int(np.floor(0.8 * len(img_ids_stem))):]

                    # Add "B" quality only to train
                    for idx in img_ids_B:
                        train_ids.append(idx.stem.split('img_')[-1])

                    train_val_ids = {'train': train_ids, 'val': val_ids}
                    with open(Path.cwd() / "segmentation" / "training" / "splits" / 'ids_{}_{}.json'.format(cell_type, mode),
                              'w', encoding='utf-8') as outfile:
                        json.dump(train_val_ids, outfile, ensure_ascii=False, indent=2)

                for train_mode in ['train', 'val']:
                    for idx in train_val_ids[train_mode]:
                        source_path = path_train_sets / "{}_{}".format(cell_type, mode)
                        target_path = path_train_sets / "{}_{}".format(cell_type, mode) / train_mode
                        if (source_path / "A" / ("img_{}.tif".format(idx))).exists():
                            source_path = source_path / "A"
                        else:
                            source_path = source_path / "B"
                        shutil.copyfile(str(source_path / "img_{}.tif".format(idx)),
                                        str(target_path / "img_{}.tif".format(idx)))
                        shutil.copyfile(str(source_path / "dist_cell_{}.tif".format(idx)),
                                        str(target_path / "dist_cell_{}.tif".format(idx)))
                        shutil.copyfile(str(source_path / "dist_neighbor_{}.tif".format(idx)),
                                        str(target_path / "dist_neighbor_{}.tif".format(idx)))
                        shutil.copyfile(str(source_path / "mask_{}.tif".format(idx)),
                                        str(target_path / "mask_{}.tif".format(idx)))