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


def copy_best_model(path_models, path_best_models, path_ctc_data, path_segmentation, best_model, best_settings, mode,
                    cell_types):
    """ Copy best models to KIT-Sch-GE_2021/SW and the best (training data set) results.

    :param path_models: Path to all saved models.
        :type path_models: pathlib Path object.
    :param path_best_models: Path to the best models.
        :type path_best_models: pathlib Path object.
    :param path_ctc_data: Path to the Cell Tracking Challenge data
        :type path_ctc_data: pathlib Path object.
    :param path_segmentation: Path to the segmentation results.
        :type path_segmentation: pathlib Path object.
    :param best_model: Name of the best model.
        :type best_model: str
    :param best_settings: Best post-processing settings (th_cell, th_mask, ...)
        :type best_settings: dict
    :param mode: Primary Track mode ('GT', 'ST', 'GT+ST', 'allGT', 'allST', 'allGT+allST')
        :type mode: str
    :param cell_types: cell types belonging to the best model.
        :type cell_types: list
    :return: None.
    """

    path_best_models.mkdir(parents=True, exist_ok=True)

    new_model_name = best_model.split(mode)[0] + mode

    # Copy and rename model
    shutil.copy(str(path_models / "{}.pth".format(best_model)),
                str(path_best_models / "{}_model.pth".format(new_model_name)))
    shutil.copy(str(path_models / "{}.json".format(best_model)),
                str(path_best_models / "{}_model.json".format(new_model_name)))

    # Copy train results
    th_seed, th_cell = best_settings['th_seed'], best_settings['th_cell']
    for cell_type in cell_types:
        save_path = path_ctc_data / 'training_datasets' / cell_type / 'KIT-Sch-GE_2021' / mode
        if save_path.exists():
            shutil.rmtree(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
        if best_settings['apply_clahe']:
            from_path = path_segmentation / mode / best_model / cell_type / "train_{}_{}_clahe".format(int(th_seed * 100),
                                                                                                       int(th_cell * 100))
        else:
            from_path = path_segmentation / mode / best_model / cell_type / "train_{}_{}".format(int(th_seed * 100),
                                                                                                 int(th_cell * 100))
        shutil.copytree(str(from_path),
                        str(path_ctc_data / 'training_datasets' / cell_type / 'KIT-Sch-GE_2021' / mode / 'CSB'))

    # Add best settings to model info file
    with open(path_best_models / "{}_model.json".format(new_model_name)) as f:
        settings = json.load(f)
    settings['best_settings'] = best_settings

    with open(path_best_models / "{}_model.json".format(new_model_name), 'w', encoding='utf-8') as outfile:
        json.dump(settings, outfile, ensure_ascii=False, indent=2)

    return None


def get_best_model(metric_scores, mode, th_cells, th_seeds):
    """ Get best model and corresponding settings.

    :param metric_scores: Scores of corresponding models.
        :type metric_scores: dict
    :param mode: Primary Track mode ('GT', 'ST', 'GT+ST', 'allGT', 'allST', 'allGT+allST')
        :type mode: str
    :param th_cells: Cell/mask thresholds which are evaluated
        :type th_cells: list
    :param th_seeds: Seed/marker thresholds which are evaluated.
        :type th_seeds: list
    :return:
    """

    best_th_cell, best_th_seed, best_model = 0, 0, ''

    if "all" in mode:

        best_op_csb = 0

        for model in metric_scores:

            for th_seed in th_seeds:

                for th_cell in th_cells:

                    op_csb = 0

                    for cell_type in metric_scores[model]:

                        if cell_type in ['Fluo-C2DL-MSC', 'Fluo-C3DH-H157']:  # makes no sense without scaling
                            continue

                        for train_set in ['01', '02']:
                            op_csb += metric_scores[model][cell_type][train_set][str(th_seed)][str(th_cell)]['OP_CSB']

                    op_csb /= len(metric_scores[model]) * 2

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

                        for train_set in ['01', '02']:

                            op_csb += metric_scores[model][cell_type][train_set][str(th_seed)][str(th_cell)]['OP_CSB']

                        op_csb /= 2

                        if op_csb > best_op_csb:
                            best_op_csb = op_csb
                            best_th_cell = th_cell
                            best_th_seed = th_seed
                            best_model = model

    return best_op_csb, best_th_cell, best_th_seed, best_model


def zero_pad_model_input(img, pad_val=0):
    """ Zero-pad model input to get for the model needed sizes (more intelligent padding ways could easily be
        implemented but there are sometimes cudnn errors with image sizes which work on cpu ...).

    :param img: Model input image.
        :type:
    :param pad_val: Value to pad.
        :type pad_val: int.

    :return: (zero-)padded img, [0s padded in y-direction, 0s padded in x-direction]
    """

    if len(img.shape) == 3:  # 3D image (z-dimension needs no pads)
        img = np.transpose(img, (2, 1, 0))

    if img.shape[0] < 64:
        # Zero-pad to 128
        y_pads = 64 - img.shape[0]

    elif img.shape[0] == 64:
        # No zero-padding needed
        y_pads = 0

    elif img.shape[0] < 128:
        # Zero-pad to 128
        y_pads = 128 - img.shape[0]

    elif img.shape[0] == 128:
        # No zero-padding needed
        y_pads = 0

    elif 128 < img.shape[0] < 256:
        # Zero-pad to 256
        y_pads = 256 - img.shape[0]

    elif img.shape[0] == 256:
        # No zero-padding needed
        y_pads = 0

    elif 256 < img.shape[0] < 512:
        # Zero-pad to 512
        y_pads = 512 - img.shape[0]

    elif img.shape[0] == 512:
        # No zero-padding needed
        y_pads = 0

    elif 512 < img.shape[0] < 768:
        # Zero-pad to 768
        y_pads = 768 - img.shape[0]

    elif img.shape[0] == 768:
        # No zero-padding needed
        y_pads = 0

    elif 768 < img.shape[0] < 1024:
        # Zero-pad to 1024
        y_pads = 1024 - img.shape[0]

    elif img.shape[0] == 1024:
        # No zero-padding needed
        y_pads = 0

    elif 1024 < img.shape[0] < 1280:
        # Zero-pad to 2048
        y_pads = 1280 - img.shape[0]

    elif img.shape[0] == 1280:
        # No zero-padding needed
        y_pads = 0

    elif 1280 < img.shape[0] < 1920:
        # Zero-pad to 1920
        y_pads = 1920 - img.shape[0]

    elif img.shape[0] == 1920:
        # No zero-padding needed
        y_pads = 0

    elif 1920 < img.shape[0] < 2048:
        # Zero-pad to 2048
        y_pads = 2048 - img.shape[0]

    elif img.shape[0] == 2048:
        # No zero-padding needed
        y_pads = 0

    elif 2048 < img.shape[0] < 2560:
        # Zero-pad to 2560
        y_pads = 2560 - img.shape[0]

    elif img.shape[0] == 2560:
        # No zero-padding needed
        y_pads = 0

    elif 2560 < img.shape[0] < 4096:
        # Zero-pad to 4096
        y_pads = 4096 - img.shape[0]

    elif img.shape[0] == 4096:
        # No zero-padding needed
        y_pads = 0

    elif 4096 < img.shape[0] < 8192:
        # Zero-pad to 8192
        y_pads = 8192 - img.shape[0]

    elif img.shape[0] == 8192:
        # No zero-padding needed
        y_pads = 0
    else:
        raise Exception('Padding error. Image too big?')

    if img.shape[1] < 64:
        # Zero-pad to 128
        x_pads = 64 - img.shape[1]

    elif img.shape[1] == 64:
        # No zero-padding needed
        x_pads = 0

    elif img.shape[1] < 128:
        # Zero-pad to 128
        x_pads = 128 - img.shape[1]

    elif img.shape[1] == 128:
        # No zero-padding needed
        x_pads = 0

    elif 128 < img.shape[1] < 256:
        # Zero-pad to 256
        x_pads = 256 - img.shape[1]

    elif img.shape[1] == 256:
        # No zero-padding needed
        x_pads = 0

    elif 256 < img.shape[1] < 512:
        # Zero-pad to 512
        x_pads = 512 - img.shape[1]

    elif img.shape[1] == 512:
        # No zero-padding needed
        x_pads = 0

    elif 512 < img.shape[1] < 768:
        # Zero-pad to 768
        x_pads = 768 - img.shape[1]

    elif img.shape[1] == 768:
        # No zero-padding needed
        x_pads = 0

    elif 768 < img.shape[1] < 1024:
        # Zero-pad to 1024
        x_pads = 1024 - img.shape[1]

    elif img.shape[1] == 1024:
        # No zero-padding needed
        x_pads = 0

    elif 1024 < img.shape[1] < 1280:
        # Zero-pad to 1024
        x_pads = 1280 - img.shape[1]

    elif img.shape[1] == 1280:
        # No zero-padding needed
        x_pads = 0

    elif 1280 < img.shape[1] < 1920:
        # Zero-pad to 1920
        x_pads = 1920 - img.shape[1]

    elif img.shape[1] == 1920:
        # No zero-padding needed
        x_pads = 0

    elif 1920 < img.shape[1] < 2048:
        # Zero-pad to 2048
        x_pads = 2048 - img.shape[1]

    elif img.shape[1] == 2048:
        # No zero-padding needed
        x_pads = 0

    elif 2048 < img.shape[1] < 2560:
        # Zero-pad to 2560
        x_pads = 2560 - img.shape[1]

    elif img.shape[1] == 2560:
        # No zero-padding needed
        x_pads = 0

    elif 2560 < img.shape[1] < 4096:
        # Zero-pad to 4096
        x_pads = 4096 - img.shape[1]

    elif img.shape[1] == 4096:
        # No zero-padding needed
        x_pads = 0

    elif 4096 < img.shape[1] < 8192:
        # Zero-pad to 8192
        x_pads = 8192 - img.shape[1]

    elif img.shape[1] == 8192:
        # No zero-padding needed
        x_pads = 0
    else:
        raise Exception('Padding error. Image too big?')

    if len(img.shape) == 3:  # 3D image
        img = np.pad(img, ((y_pads, 0), (x_pads, 0), (0, 0)), mode='constant', constant_values=pad_val)
        img = np.transpose(img, (2, 1, 0))
    else:
        img = np.pad(img, ((y_pads, 0), (x_pads, 0)), mode='constant', constant_values=pad_val)

    return img, [y_pads, x_pads]
