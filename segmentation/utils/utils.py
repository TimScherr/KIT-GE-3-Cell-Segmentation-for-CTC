import json
import numpy as np
import shutil


def get_det_score(path):
    """  Get DET metric score.

    :param path: path to the DET score file.
        :type path: pathlib Path object.
    :return: DET score.
    """

    with open(path) as det_file:
        read_file = True
        while read_file:
            det_file_line = det_file.readline()
            if 'DET' in det_file_line:
                det_score = float(det_file_line.split('DET measure: ')[-1].split('\n')[0])
                read_file = False

    return det_score


def get_seg_score(path):
    """  Get SEG metric score.

    :param path: path to the SEG score file.
        :type path: pathlib Path object.
    :return: SEG score.
    """

    with open(path) as det_file:
        read_file = True
        while read_file:
            det_file_line = det_file.readline()
            if 'SEG' in det_file_line:
                seg_score = float(det_file_line.split('SEG measure: ')[-1].split('\n')[0])
                read_file = False

    return seg_score


def get_nucleus_ids(img):
    """ Get nucleus ids in intensity-coded label image.

    :param img: Intensity-coded nuclei image.
        :type:
    :return: List of nucleus ids.
    """

    values = np.unique(img)
    values = values[values > 0]

    return values


def min_max_normalization(img, min_value=None, max_value=None):
    """ Minimum maximum normalization.

    :param img: Image (uint8, uint16 or int)
        :type img:
    :param min_value: minimum value for normalization, values below are clipped.
        :type min_value: int
    :param max_value: maximum value for normalization, values above are clipped.
        :type max_value: int
    :return: Normalized image (float32)
    """

    if max_value is None:
        max_value = img.max()

    if min_value is None:
        min_value = img.min()

    # Clip image to filter hot and cold pixels
    img = np.clip(img, min_value, max_value)

    # Apply min-max-normalization
    img = 2 * (img.astype(np.float32) - min_value) / (max_value - min_value) - 1

    return img.astype(np.float32)


def unique_path(directory, name_pattern):
    """ Get unique file name to save trained model.

    :param directory: Path to the model directory
        :type directory: pathlib path object.
    :param name_pattern: Pattern for the file name
        :type name_pattern: str
    :return: pathlib path
    """
    counter = 0
    while True:
        counter += 1
        path = directory / name_pattern.format(counter)
        if not path.exists():
            return path


def write_train_info(configs, path):
    """ Write training configurations into a json file.

    :param configs: Dictionary with configurations of the training process.
        :type configs: dict
    :param path: path to the directory to store the json file.
        :type path: pathlib Path object
    :return: None
    """

    with open(path / (configs['run_name'] + '.json'), 'w', encoding='utf-8') as outfile:
        json.dump(configs, outfile, ensure_ascii=False, indent=2)

    return None


def copy_best_model(path_models, path_best_models, best_model, best_settings):
    """ Copy best models to KIT-Sch-GE_2021/SW and the best (training data set) results.

    :param path_models: Path to all saved models.
        :type path_models: pathlib Path object.
    :param path_best_models: Path to the best models.
        :type path_best_models: pathlib Path object.
    :param best_model: Name of the best model.
        :type best_model: str
    :param best_settings: Best post-processing settings (th_cell, th_mask, ...)
        :type best_settings: dict
    :return: None.
    """

    new_model_name = best_model[:-3]

    # Copy and rename model
    shutil.copy(str(path_models / "{}.pth".format(best_model)),
                str(path_best_models / "{}.pth".format(new_model_name)))
    shutil.copy(str(path_models / "{}.json".format(best_model)),
                str(path_best_models / "{}.json".format(new_model_name)))

    # Add best settings to model info file
    with open(path_best_models / "{}.json".format(new_model_name)) as f:
        settings = json.load(f)
    settings['best_settings'] = best_settings

    with open(path_best_models / "{}.json".format(new_model_name), 'w', encoding='utf-8') as outfile:
        json.dump(settings, outfile, ensure_ascii=False, indent=2)

    return None


def get_best_model(metric_scores, mode, subset, th_cells, th_seeds):
    """ Get best model and corresponding settings.

    :param metric_scores: Scores of corresponding models.
        :type metric_scores: dict
    :param mode: Mode ('all', 'single')
        :type mode: str
    :param subset: Evaluate on dataset '01', '02' or on both ('01+02').
        :type subset: str
    :param th_cells: Cell/mask thresholds which are evaluated
        :type th_cells: list
    :param th_seeds: Seed/marker thresholds which are evaluated.
        :type th_seeds: list
    :return:
    """

    best_th_cell, best_th_seed, best_model = 0, 0, ''

    subsets = [subset]
    if subset == '01+02':
        subsets = ['01', '02']

    if "all" in mode:

        best_op_csb = 0

        for model in metric_scores:

            for th_seed in th_seeds:

                for th_cell in th_cells:

                    op_csb = 0

                    for cell_type in metric_scores[model]:

                        # Exclude too different cell types (goal: better model on other data sets)
                        if cell_type in ['Fluo-C2DL-MSC', 'Fluo-C3DH-H157']:
                            continue

                        for train_set in subsets:
                            op_csb += metric_scores[model][cell_type][train_set][str(th_seed)][str(th_cell)]['OP_CSB']

                    op_csb /= len(metric_scores[model]) * len(subsets)

                    if op_csb > best_op_csb:
                        best_op_csb = op_csb
                        best_th_cell = th_cell
                        best_th_seed = th_seed
                        best_model = model

    else:

        best_op_csb = 0

        for model in metric_scores:

            for cell_type in metric_scores[model]:

                for th_seed in th_seeds:

                    for th_cell in th_cells:

                        op_csb = 0

                        for train_set in subsets:

                            op_csb += metric_scores[model][cell_type][train_set][str(th_seed)][str(th_cell)]['OP_CSB']

                        op_csb /= len(subsets)

                        if op_csb > best_op_csb:
                            best_op_csb = op_csb
                            best_th_cell = th_cell
                            best_th_seed = th_seed
                            best_model = model

    return best_op_csb, float(best_th_cell), float(best_th_seed), best_model


def zero_pad_model_input(img, pad_val=0):
    """ Zero-pad model input to get for the model needed sizes (more intelligent padding ways could easily be
        implemented but there are sometimes cudnn errors with image sizes which work on cpu ...).

    :param img: Model input image.
        :type:
    :param pad_val: Value to pad.
        :type pad_val: int.

    :return: (zero-)padded img, [0s padded in y-direction, 0s padded in x-direction]
    """

    # Tested shapes
    tested_img_shapes = [64, 128, 256, 320, 512, 768, 1024, 1280, 1408, 1600, 1920, 2048, 2240, 2560, 3200, 4096,
                         4480, 6080, 8192]

    if len(img.shape) == 3:  # 3D image (z-dimension needs no pads)
        img = np.transpose(img, (2, 1, 0))

    # More effective padding (but may lead to cuda errors)
    # y_pads = int(np.ceil(img.shape[0] / 64) * 64) - img.shape[0]
    # x_pads = int(np.ceil(img.shape[1] / 64) * 64) - img.shape[1]

    pads = []
    for i in range(2):
        for tested_img_shape in tested_img_shapes:
            if img.shape[i] <= tested_img_shape:
                pads.append(tested_img_shape - img.shape[i])
                break

    if not pads:
        raise Exception('Image too big to pad. Use sliding windows')

    if len(img.shape) == 3:  # 3D image
        img = np.pad(img, ((pads[0], 0), (pads[1], 0), (0, 0)), mode='constant', constant_values=pad_val)
        img = np.transpose(img, (2, 1, 0))
    else:
        img = np.pad(img, ((pads[0], 0), (pads[1], 0)), mode='constant', constant_values=pad_val)

    return img, [pads[0], pads[1]]
