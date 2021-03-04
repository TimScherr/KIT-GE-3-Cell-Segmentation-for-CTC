import argparse
import json
import numpy as np
import random
import torch
import warnings

from pathlib import Path

from segmentation.inference.inference import inference_2d_ctc, inference_3d_ctc
from segmentation.training.cell_segmentation_dataset import CellSegDataset
from segmentation.training.autoencoder_dataset import AutoEncoderDataset
from segmentation.training.create_training_sets import create_ctc_training_sets
from segmentation.training.mytransforms import augmentors
from segmentation.training.training import train, train_auto
from segmentation.utils import utils, unets
from segmentation.utils.metrics import ctc_metrics, count_det_errors

warnings.filterwarnings("ignore", category=UserWarning)


class EvalArgs(object):
    """ Class with post-processing parameters.

    """

    def __init__(self, th_cell, th_seed, n_splitting, apply_clahe, scale, cell_type, save_raw_pred,
                 artifact_correction, fuse_z_seeds):
        """

        :param th_cell: Mask / cell size threshold.
            :type th_cell: float
        :param th_seed: Seed / marker threshold.
            :type th_seed: float
        :param n_splitting: Number of detected cells above which to apply additional splitting (only for 3D).
            :type n_splitting: int
        :param apply_clahe: Apply contrast limited adaptive histogram equalization (CLAHE).
            :type apply_clahe: bool
        :param scale: Scale factor for downsampling.
            :type scale: float
        :param cell_type: Cell type.
            :type cell_type: str
        :param save_raw_pred: Save (some) raw predictions.
            :type save_raw_pred: bool
        :param artifact_correction: Apply artifact correction post-processing.
            :type artifact_correction: bool
        :param fuse_z_seeds: Fuse seeds in z-direction / axial direction.
            :type fuse_z_seeds: bool
        """
        self.th_cell = th_cell
        self.th_seed = th_seed
        self.n_splitting = n_splitting
        self.apply_clahe = apply_clahe
        self.scale = scale
        self.cell_type = cell_type
        self.save_raw_pred = save_raw_pred
        self.artifact_correction = artifact_correction
        self.fuse_z_seeds = fuse_z_seeds


def main():

    random.seed()
    np.random.seed()

    # Get arguments
    parser = argparse.ArgumentParser(description='KIT-Sch-GE 2021 Cell Segmentation')
    parser.add_argument('--train', '-t', default=False, action='store_true', help='Train new models')
    parser.add_argument('--evaluate', '-e', default=False, action='store_true', help='Evaluate models')
    parser.add_argument('--inference', '-i', default=False, action='store_true', help='Inference')
    parser.add_argument('--save_raw_pred', '-s', default=False, action='store_true', help='Save raw predictions')
    parser.add_argument('--cell_type', '-c', default='all', type=str, help='Cell type')
    parser.add_argument('--mode', '-m', default='GT', type=str, help='Mode for training')
    parser.add_argument('--th_seed', '-th_s', default=0.45, type=float, help='Seed threshold')
    parser.add_argument('--th_cell', '-th_c', default=0.08, type=float, help='Cell size threshold')
    parser.add_argument('--apply_clahe', '-ac', default=False, action='store_true', help='Apply CLAHE')
    parser.add_argument('--n_splitting', '-ns', default=40, type=int, help='Cell number to apply local splitting (only 3D)')
    parser.add_argument('--scale', '-sc', default=1.0, type=float, help='Scale for down-/upsampling (inference)')
    parser.add_argument('--batch_size', '-bs', default=8, type=int, help='Batch size (inference)')
    parser.add_argument('--multi_gpu', '-mgpu', default=True, action='store_true', help='Use multiple GPUs')
    parser.add_argument('--artifact_correction', default=False, action='store_true', help='Artifact correction (only for very dense cells, e.g., HSC')
    parser.add_argument('--fuse_z_seeds', default=False, action='store_true', help='Fuse seeds in z-direction')
    args = parser.parse_args()

    # Load settings and paths
    with open(Path.cwd() / 'cell_segmentation_train_settings.json') as f:
        settings = json.load(f)

    with open(Path.cwd() / 'paths.json') as f:
        paths = json.load(f)

    # Paths
    path_datasets = Path(paths['path_data'])
    path_results = Path(paths['path_results'])
    if path_results == '':
        path_results = path_datasets
    path_models = path_results / 'segmentation_models'
    path_best_models = path_datasets / 'kit-sch-ge_2021' / 'SW'
    path_train_data = path_results / 'training_sets'
    path_ctc_metric = Path(paths['path_ctc_metric'])
    if args.cell_type == 'all':
        cell_types = paths['cell_types']
    else:
        cell_types = [args.cell_type]

    # Set device for using CPU or GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if str(device) == 'cuda':
        torch.backends.cudnn.benchmark = True
    if args.multi_gpu:
        num_gpus = torch.cuda.device_count()
    else:
        num_gpus = 1

    if args.train:  # Train model from scratch

        # Make directory for the trained models
        path_models.mkdir(exist_ok=True)

        print('Create training sets for all cell types ...')
        create_ctc_training_sets(path_data=path_datasets,
                                 path_train_sets=path_train_data,
                                 cell_types=paths['cell_types'])

        for cell_type in cell_types:

            for architecture in settings['methods']:

                # Get model names and how many iterations/models need to be trained
                if "all" in args.mode:
                    model_name = '{}_{}'.format(args.mode, architecture[2])
                else:
                    model_name = '{}_{}_{}'.format(cell_type, args.mode, architecture[2])
                if args.mode == 'GT' and architecture[-1]:  # auto-encoder pre-training only for GT
                    model_name += '-auto'
                num_trained_models = len(list(path_models.glob('{}_model*.pth'.format(model_name))))
                if "all" in args.mode or "ST" in args.mode:
                    iterations = settings['iterations'] - num_trained_models
                else:
                    iterations = settings['iterations_GT_single_celltype'] - num_trained_models
                    
                if iterations <= 0:
                    continue

                for i in range(iterations):  # Train multiple models

                    run_name = utils.unique_path(path_models, model_name + '_model_{:02d}.pth').stem

                    train_configs = {'architecture': architecture[0],
                                     'batch_size': settings['batch_size'],
                                     'batch_size_auto': settings['batch_size_auto'],
                                     'label_type': architecture[1],
                                     'loss': architecture[3],
                                     'num_gpus': num_gpus,
                                     'optimizer': architecture[2],
                                     'run_name': run_name
                                     }

                    net = unets.build_unet(unet_type=train_configs['architecture'][0],
                                           act_fun=train_configs['architecture'][2],
                                           pool_method=train_configs['architecture'][1],
                                           normalization=train_configs['architecture'][3],
                                           device=device,
                                           num_gpus=num_gpus,
                                           ch_in=1,
                                           ch_out=1,
                                           filters=train_configs['architecture'][4])

                    if 'auto' in run_name:  # Pre-training of the Encoder
                        net_auto = unets.build_unet(unet_type='AutoU',
                                                    act_fun=train_configs['architecture'][2],
                                                    pool_method=train_configs['architecture'][1],
                                                    normalization=train_configs['architecture'][3],
                                                    device=device,
                                                    num_gpus=num_gpus,
                                                    ch_in=1,
                                                    ch_out=1,
                                                    filters=train_configs['architecture'][4])

                        data_transforms_auto = augmentors(label_type='auto', min_value=0, max_value=65535)

                        # Load training and validation set
                        datasets = AutoEncoderDataset(root_dir=path_datasets,
                                                      cell_type=cell_type,
                                                      gt_train_dir= path_results / 'training_sets',
                                                      transform=data_transforms_auto)

                        # Train model
                        train_auto(net=net_auto,
                                   dataset=datasets,
                                   configs=train_configs,
                                   device=device,
                                   path_models=path_models)

                        # Load weights
                        if num_gpus > 1:
                            net_auto.module.load_state_dict(torch.load(str(path_models / '{}.pth'.format(run_name)),
                                                                       map_location=device))
                        else:
                            net_auto.load_state_dict(torch.load(str(path_models / '{}.pth'.format(run_name)),
                                                                map_location=device))

                        pretrained_dict = net_auto.state_dict()
                        net_dict = net.state_dict()

                        # 1. filter out unnecessary keys
                        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in net_dict}
                        # 2. overwrite entries in the existing state dict
                        net_dict.update(pretrained_dict)
                        # 3. load the new state dict
                        net.load_state_dict(net_dict)

                        del net_auto

                    # The training images are uint16 crops of a min-max normalized image
                    data_transforms = augmentors(label_type=train_configs['label_type'], min_value=0, max_value=65535)
                    train_configs['data_transforms'] = str(data_transforms)

                    # Load training and validation set
                    datasets = {x: CellSegDataset(root_dir=path_results / 'training_sets',
                                                  mode=x,
                                                  cell_type=cell_type,
                                                  gt_mode=args.mode,
                                                  transform=data_transforms[x])
                                for x in ['train', 'val']}

                    # Get number of training epochs depending on dataset size (just roughly to decrease training time):
                    if len(datasets['train']) + len(datasets['val']) >= 1000:
                        train_configs['max_epochs'] = 200
                    elif len(datasets['train']) + len(datasets['val']) >= 500:
                        train_configs['max_epochs'] = 240
                    elif len(datasets['train']) + len(datasets['val']) >= 200:
                        train_configs['max_epochs'] = 320
                    elif len(datasets['train']) + len(datasets['val']) >= 100:
                        train_configs['max_epochs'] = 400
                    elif len(datasets['train']) + len(datasets['val']) >= 50:
                        train_configs['max_epochs'] = 480
                    else:
                        train_configs['max_epochs'] = 560

                    # Train model
                    best_loss = train(net=net,
                                      datasets=datasets,
                                      configs=train_configs,
                                      device=device,
                                      path_models=path_models)

                    if train_configs['optimizer'] == 'ranger':  # Fine-tune with cosine annealing
                        net = unets.build_unet(unet_type=train_configs['architecture'][0],
                                               act_fun=train_configs['architecture'][2],
                                               pool_method=train_configs['architecture'][1],
                                               normalization=train_configs['architecture'][3],
                                               device=device,
                                               num_gpus=num_gpus,
                                               ch_in=1,
                                               ch_out=1,
                                               filters=train_configs['architecture'][4])

                        # Get best weights as starting point
                        if num_gpus > 1:
                            net.module.load_state_dict(torch.load(str(path_models / '{}.pth'.format(run_name)),
                                                                  map_location=device))
                        else:
                            net.load_state_dict(torch.load(str(path_models / '{}.pth'.format(run_name)),
                                                           map_location=device))
                        _ = train(net=net,
                                  datasets=datasets,
                                  configs=train_configs,
                                  device=device,
                                  path_models=path_models,
                                  best_loss=best_loss)

                    # Write information to json-file
                    utils.write_train_info(configs=train_configs, path=path_models)

    if args.evaluate:  # Eval all trained models on the test data set

        if not path_ctc_metric.is_dir():
            return print('No CTC evaluation software found. Download the software'
                         '(http://celltrackingchallenge.net/evaluation-methodology/)\n')

        # Make results directory
        Path.joinpath(path_results, 'segmentation').mkdir(exist_ok=True)
        (path_results / 'segmentation' / args.mode).mkdir(exist_ok=True)

        # Get models and cell types to evaluate
        model_eval_list = []
        if "all" in args.mode:
            for architecture in settings['methods']:
                model_name = '{}_{}'.format(args.mode, architecture[2])
                if architecture[-1]:  # auto-encoder pre-training only for GT
                    continue
                models = sorted(path_models.glob('{}_model*.pth'.format(model_name)))
                for model in models:
                    model_eval_list.append([model, cell_types])
        else:
            for cell_type in cell_types:
                for architecture in settings['methods']:
                    model_name = '{}_{}_{}'.format(cell_type, args.mode, architecture[2])
                    if architecture[-1] and args.mode != 'GT':  # auto-encoder pre-training only for GT
                        continue
                    elif architecture[-1] and args.mode == 'GT':
                        model_name += '-auto'
                    models = sorted(path_models.glob('{}_model*.pth'.format(model_name)))
                    for model in models:
                        model_eval_list.append([model, [cell_type]])

        # Go through model list and evaluate for needed cell_types
        metric_scores = {}
        for model, cell_type_list in model_eval_list:

            metric_scores[model.stem] = {}
            (path_results / 'segmentation' / args.mode / model.stem).mkdir(exist_ok=True)

            for cell_type in cell_type_list:

                metric_scores[model.stem][cell_type] = {'01': {}, '02': {}}
                (path_results / 'segmentation' / args.mode / model.stem / cell_type).mkdir(exist_ok=True)

                for train_set in ['01', '02']:

                    metric_scores[model.stem][cell_type][train_set] = {}

                    if ('all' in args.mode) or ('ST' in args.mode):  # too time consuming to test more
                        th_seeds = [0.35, 0.45]  # seed thresholds to test
                        th_cells = [0.07]  # cell thresholds to test
                        scale_factor = 1
                    else:
                        if cell_type == 'Fluo-N3DH-CE':  # takes too much time to evaluate
                            th_seeds = [0.35]  # seed thresholds to test
                            th_cells = [0.07]  # cell thresholds to test
                        else:
                            th_seeds = [0.35, 0.45]  # seed thresholds to test
                            th_cells = [0.07, 0.09]  # cell thresholds to test

                        with open(path_train_data / "{}_{}".format(cell_type, args.mode) / 'info.json') as f:
                            scale_factor = json.load(f)["scale"]

                    for th_seed in th_seeds:

                        metric_scores[model.stem][cell_type][train_set][str(th_seed)] = {}

                        for th_cell in th_cells:

                            metric_scores[model.stem][cell_type][train_set][str(th_seed)][str(th_cell)] = {}

                            print('Evaluate {} on {}_{}: th_seed: {}, th_cell: {}'.format(model.stem,
                                                                                          cell_type,
                                                                                          train_set,
                                                                                          th_seed,
                                                                                          th_cell))
                            path_seg_results = path_results / 'segmentation' / args.mode / model.stem / cell_type
                            if args.apply_clahe:
                                path_seg_results = path_seg_results / "train_{}_{}_clahe".format(int(th_seed * 100),
                                                                                                 int(th_cell * 100)) / '{}_RES'.format(train_set)
                            else:
                                path_seg_results = path_seg_results / "train_{}_{}".format(int(th_seed * 100),
                                                                                                 int(th_cell * 100)) / '{}_RES'.format(train_set)

                            path_seg_results.mkdir(exist_ok=True, parents=True)

                            if (path_seg_results / "SEG_log.txt").exists():
                                if args.mode in ['ST', 'allST']:
                                    det_measure, so, fnv, fpv = 0, np.nan, np.nan, np.nan
                                    seg_measure = utils.get_seg_score(path_seg_results / "SEG_log_ST.txt")
                                else:
                                    det_measure, so, fnv, fpv = count_det_errors(path_seg_results / "DET_log.txt")
                                    seg_measure = utils.get_seg_score(path_seg_results / "SEG_log.txt")
                                op_csb = (det_measure + seg_measure) / 2
                                metric_scores[model.stem][cell_type][train_set][str(th_seed)][str(th_cell)]['DET'] = det_measure
                                metric_scores[model.stem][cell_type][train_set][str(th_seed)][str(th_cell)]['SEG'] = seg_measure
                                metric_scores[model.stem][cell_type][train_set][str(th_seed)][str(th_cell)]['OP_CSB'] = op_csb
                                metric_scores[model.stem][cell_type][train_set][str(th_seed)][str(th_cell)]['SO'] = so
                                metric_scores[model.stem][cell_type][train_set][str(th_seed)][str(th_cell)]['FPV'] = fpv
                                metric_scores[model.stem][cell_type][train_set][str(th_seed)][str(th_cell)]['FNV'] = fnv

                                continue

                            eval_args = EvalArgs(th_cell=th_cell, th_seed=th_seed, n_splitting=args.n_splitting,
                                                 apply_clahe=args.apply_clahe, scale=scale_factor, cell_type=cell_type,
                                                 save_raw_pred=args.save_raw_pred,
                                                 artifact_correction=args.artifact_correction,
                                                 fuse_z_seeds=args.fuse_z_seeds)

                            if '2D' in cell_type:
                                inference_2d_ctc(model=model,
                                                 data_path=path_datasets / 'training_datasets' / cell_type / train_set,
                                                 result_path=path_seg_results,
                                                 device=device,
                                                 batchsize=args.batch_size,
                                                 args=eval_args,
                                                 num_gpus=num_gpus)
                            else:
                                inference_3d_ctc(model=model,
                                                 data_path=path_datasets / 'training_datasets' / cell_type / train_set,
                                                 result_path=path_seg_results,
                                                 device=device,
                                                 batchsize=args.batch_size,
                                                 args=eval_args,
                                                 num_gpus=num_gpus)

                            seg_measure, det_measure = ctc_metrics(path_data=path_datasets / 'training_datasets' / cell_type,
                                                                   path_results=path_seg_results,
                                                                   path_software=path_ctc_metric,
                                                                   subset=train_set,
                                                                   mode=args.mode)
                            if args.mode in ['ST', 'allST']:  # Calculate also GT metrics for further analysis
                                _, _ = ctc_metrics(path_data=path_datasets / 'training_datasets' / cell_type,
                                                   path_results=path_seg_results,
                                                   path_software=path_ctc_metric,
                                                   subset=train_set,
                                                   mode='GT')

                            if args.mode in ['ST', 'allST']:
                                so, fnv, fpv = np.nan, np.nan, np.nan
                            else:
                                _, so, fnv, fpv = count_det_errors(path_seg_results / "DET_log.txt")
                            op_csb = (det_measure + seg_measure) / 2
                            metric_scores[model.stem][cell_type][train_set][str(th_seed)][str(th_cell)]['DET'] = det_measure
                            metric_scores[model.stem][cell_type][train_set][str(th_seed)][str(th_cell)]['SEG'] = seg_measure
                            metric_scores[model.stem][cell_type][train_set][str(th_seed)][str(th_cell)]['OP_CSB'] = op_csb
                            metric_scores[model.stem][cell_type][train_set][str(th_seed)][str(th_cell)]['SO'] = so
                            metric_scores[model.stem][cell_type][train_set][str(th_seed)][str(th_cell)]['FPV'] = fpv
                            metric_scores[model.stem][cell_type][train_set][str(th_seed)][str(th_cell)]['FNV'] = fnv

        # Save evaluation metric scores --> clahe in file_name --> correct dir all vs not all ...
        if "all" in args.mode:
            cell_type = "all"

        if args.apply_clahe:
            flag = '_clahe'
        else:
            flag = ''

        with open(path_results / 'segmentation' / args.mode / 'metrics_{}_{}{}.json'.format(cell_type,
                                                                                            args.mode,
                                                                                            flag),
                  'w', encoding='utf-8') as outfile:
            json.dump(metric_scores, outfile, ensure_ascii=False, indent=2)

        # Get best model
        best_op_csb, best_th_cell, best_th_seed, best_model = utils.get_best_model(metric_scores=metric_scores,
                                                                                   mode=args.mode,
                                                                                   th_cells=th_cells,
                                                                                   th_seeds=th_seeds)
        best_settings = {'th_cell': best_th_cell,
                         'th_seed': best_th_seed,
                         'scale_factor': scale_factor,
                         'apply_clahe': args.apply_clahe,
                         'OP_CSB': best_op_csb}

        if args.apply_clahe:
            if (path_results / 'segmentation' / args.mode / 'metrics_{}_{}.json'.format(cell_type, args.mode)).exists():
                with open(path_results / 'segmentation' / args.mode / 'metrics_{}_{}.json'.format(cell_type, args.mode)) as f:
                    metric_scores_wo = json.load(f)
                best_op_csb_wo, _, _, _ = utils.get_best_model(metric_scores=metric_scores_wo,
                                                               mode=args.mode,
                                                               th_cells=th_cells,
                                                               th_seeds=th_seeds)
            else:
                best_op_csb_wo = 0

        else:
            best_op_csb_wo = 0

        if best_op_csb - best_op_csb_wo > 0.005:

            utils.copy_best_model(path_models=path_models,
                                  path_best_models=path_best_models,
                                  path_ctc_data=path_datasets,
                                  path_segmentation=path_results / 'segmentation',
                                  best_model=best_model,
                                  best_settings=best_settings,
                                  mode=args.mode,
                                  cell_types=cell_types)

    if args.inference:  # Use best model to make predictions on the challenge datasets

        for cell_type in cell_types:

            if "all" in args.mode:
                model = (path_best_models / "{}_model.pth".format(args.mode))
            else:
                model = (path_best_models / "{}_{}_model.pth".format(cell_type, args.mode))

            for subset in ['01', '02']:

                path_seg_results = path_datasets / 'challenge_datasets' / cell_type / 'KIT-Sch-GE_2021' / args.mode\
                                   / 'CSB' / "{}_RES".format(subset)
                path_seg_results.mkdir(parents=True, exist_ok=True)

                print('Inference using {} on {}_{}: th_seed: {}, th_cell: {}, scale: {}, clahe:{}'.format(model.stem,
                                                                                                          cell_type,
                                                                                                          subset,
                                                                                                          args.th_seed,
                                                                                                          args.th_cell,
                                                                                                          args.scale,
                                                                                                          args.apply_clahe))

                if args.fuse_z_seeds:
                    print('Seed fusion in z-direction: on')

                if len(sorted(path_seg_results.glob('*.tif'))) > 0:
                    continue

                if '2D' in cell_type:
                    inference_2d_ctc(model=model,
                                     data_path=path_datasets / 'challenge_datasets' / cell_type / subset,
                                     result_path=path_seg_results,
                                     device=device,
                                     batchsize=args.batch_size,
                                     args=args,
                                     num_gpus=num_gpus)
                else:
                    inference_3d_ctc(model=model,
                                     data_path=path_datasets / 'challenge_datasets' / cell_type / subset,
                                     result_path=path_seg_results,
                                     device=device,
                                     batchsize=args.batch_size,
                                     args=args,
                                     num_gpus=num_gpus)


if __name__ == "__main__":

    main()
