eval "$(conda shell.bash hook)"
conda activate /srv/scherr/virtual_environments/kit-sch-ge-2021-cell_segmentation_ve
python ./cell_segmentation.py --train --cell_type "Fluo-N3DH-SIM+" --mode "GT"
python ./cell_segmentation.py --evaluate --cell_type "Fluo-N3DH-SIM+" --mode "GT" --save_raw_pred --batch_size 8 --fuse_z_seeds --n_splitting 100
conda deactivate

