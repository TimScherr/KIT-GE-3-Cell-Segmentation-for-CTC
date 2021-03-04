# KIT-Sch-GE 2021 Segmentation

Segmentation method used for our submission to the 6th edition of the [ISBI Cell Tracking Challenge](http://celltrackingchallenge.net/) 2021 (Team KIT-Sch-GE).

## Prerequisites
* [Anaconda Distribution](https://www.anaconda.com/products/individual)
* A CUDA capable GPU
* Minimum / recommended RAM: 16 GiB / 32 GiB
* Minimum / recommended VRAM: 12 GiB / 24 GiB

## Installation
Clone the repository:
```
git clone https://git.scc.kit.edu/KIT-Sch-GE/2021_segmentation.git
```
Open the Anaconda Prompt (Windows) or the Terminal (Linux), go to the repository and create a new virtual environment:
```
cd path_to_the_cloned_repository
conda env create -f requirements.yml
```
Activate the virtual environment kit_sch-ge-2021_cell_segmentation_ve:
```
conda activate kit_sch-ge-2021_cell_segmentation_ve
```

## Cell Tracking Challenge 2021
In this section, it is described how to reproduce the segmentation results of our Cell Tracking Challenge submission. If the exact submission results are needed, download our trained models from the Cell Tracking Challenge website when available (and move them to *cell_tracking_challenge/kit-sch-ge_2021/SW/models*, see also next step).
 
### Data
Download the Cell Tracking Challenge training and challenge data sets. Make a folder *cell_tracking_challenge*. Unzip the training data sets into *cell_tracking_challenge/training_datasets*. Unzip the training data sets into *cell_tracking_challenge/challenge_datasets*. Download and unzip the [evaluation software](http://public.celltrackingchallenge.net/software/EvaluationSoftware.zip). Set the corresponding paths in *paths.json*.

### Training
After downloading the required Cell Tracking Challenge data, new models can be trained with:
```
python cell_segmentation.py --train --cell_type "cell_type" --mode "mode"
```
Thereby, the needed training data will be created automatically using the train/val splits in *2021_segmentation/segmentation/training/splits* (takes some time). To use new random splits, just delete all json files in the corresponding folder (and the training sets if already created).

The batch size and how many models are trained per *cell_type* and *mode* (GT, ST, GT+ST, allGT, allST, allGT+allST, depending on which label type should be used) can be adjusted in *cell_segmentation_train_settings.json*. With the standard setting a model is trained with the Adam optimizer and a model with the [Ranger](https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer) optimizer. For the mode "GT", two models are trained each and two Ranger models with an autoencoder pre-training of the encoder.

*train_eval.sh* is a bash script for the training and evaluation of our whole submission (takes some time!). 

### Evaluation
Trained models can be evaluated on the training datasets with:
```
python cell_segmentation.py --evaluate  --cell_type "cell_type" --mode "mode"
```
Some raw predictions can be saved with *--save_raw_pred*. The batch size can be set with *--batch_size $int*. For some cell types an artifact correction (*--artifact_correction*) or the fusion of seeds (in *z* direction, --fuse_z_seeds) can be helpful. The mask and marker thresholds to be evaluated can be found in *cell_segmentation.py*. For the settings ST and allST the SEG score calculated on the provided STs is used to find the best model. For the other cases, the OP_CSB is used on the provided GT data.

The best models are copied to *cell_tracking_challenge/kit-sch-ge_2021/SW/models*. In the corresponding json files, the best thresholds and the applied scaling factor can be found (and also some other information). The results of the best model are copied to *cell_tracking_challenge/training_datasets/cell_type/Kit-Sch-GE_2021/mode/csb/*. The other results can be found in the specified result_path.

### Inference
For inference, use the copied best models and the corresponding parameters:
```
python cell_segmentation.py --inference --cell_type "cell_type" --mode "mode" --save_raw_pred --batch_size $int --th_cell $float --th_seed $float (--artifact_correction --fuse_z_seeds --apply_clahe --scale $float --multi_gpu)
```
*inference.sh* is a bash script with the parameters we used for our submission (use also our trained models).

## Publication ##
T. Scherr, K. Löffler, M. Böhland, and R. Mikut (2020). Cell Segmentation and Tracking using CNN-Based Distance Predictions and a Graph-Based Matching Strategy. PLoS ONE 15(12). DOI: [10.1371/journal.pone.0243219](https://doi.org/10.1371/journal.pone.0243219).

## License ##
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.