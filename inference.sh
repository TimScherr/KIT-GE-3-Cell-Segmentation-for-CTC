eval "$(conda shell.bash hook)"

conda activate kit-sch-ge-2021-cell_segmentation_ve

python ./cell_segmentation.py --inference --cell_type "BF-C2DL-HSC" --mode "GT" --save_raw_pred --batch_size 16 --th_cell 0.07 --th_seed 0.35 --artifact_correction
python ./cell_segmentation.py --inference --cell_type "BF-C2DL-MuSC" --mode "GT" --save_raw_pred --batch_size 8 --th_cell 0.07 --th_seed 0.35 --artifact_correction
python ./cell_segmentation.py --inference --cell_type "DIC-C2DH-HeLa" --mode "GT" --save_raw_pred --batch_size 16 --th_cell 0.07 --th_seed 0.35
python ./cell_segmentation.py --inference --cell_type "Fluo-C2DL-MSC" --mode "GT" --save_raw_pred --batch_size 16 --th_cell 0.07 --th_seed 0.35 --scale 0.5
python ./cell_segmentation.py --inference --cell_type "Fluo-C3DH-A549" --mode "GT" --save_raw_pred --batch_size 8 --th_cell 0.07 --th_seed 0.35 --fuse_z_seeds
python ./cell_segmentation.py --inference --cell_type "Fluo-C3DH-H157" --mode "GT" --save_raw_pred --batch_size 8 --th_cell 0.07 --th_seed 0.35 --scale 0.6 --fuse_z_seeds
python ./cell_segmentation.py --inference --cell_type "Fluo-C3DL-MDA231" --mode "GT" --save_raw_pred --batch_size 8 --th_cell 0.07 --th_seed 0.45 --fuse_z_seeds
python ./cell_segmentation.py --inference --cell_type "Fluo-N2DH-GOWT1" --mode "GT" --save_raw_pred --batch_size 16 --th_cell 0.07 --th_seed 0.35
python ./cell_segmentation.py --inference --cell_type "Fluo-N2DL-HeLa" --mode "GT" --save_raw_pred --batch_size 16 --th_cell 0.07 --th_seed 0.35
python ./cell_segmentation.py --inference --cell_type "Fluo-N3DH-CE" --mode "GT" --save_raw_pred --batch_size 8 --th_cell 0.07 --th_seed 0.35
python ./cell_segmentation.py --inference --cell_type "Fluo-N3DH-CHO" --mode "GT" --save_raw_pred --batch_size 8 --th_cell 0.07 --th_seed 0.45  --fuse_z_seeds
python ./cell_segmentation.py --inference --cell_type "PhC-C2DH-U373" --mode "GT" --save_raw_pred --batch_size 8 --th_cell 0.07 --th_seed 0.45
python ./cell_segmentation.py --inference --cell_type "PhC-C2DL-PSC" --mode "GT" --save_raw_pred --batch_size 16 --th_cell 0.09 --th_seed 0.35

python ./cell_segmentation.py --inference --cell_type "BF-C2DL-HSC" --mode "allGT" --save_raw_pred --batch_size 16 --th_cell 0.07 --th_seed 0.35
python ./cell_segmentation.py --inference --cell_type "BF-C2DL-MuSC" --mode "allGT" --save_raw_pred --batch_size 8 --th_cell 0.07 --th_seed 0.35
python ./cell_segmentation.py --inference --cell_type "DIC-C2DH-HeLa" --mode "allGT" --save_raw_pred --batch_size 16 --th_cell 0.07 --th_seed 0.35
python ./cell_segmentation.py --inference --cell_type "Fluo-C2DL-MSC" --mode "allGT" --save_raw_pred --batch_size 16 --th_cell 0.07 --th_seed 0.35
python ./cell_segmentation.py --inference --cell_type "Fluo-C3DH-A549" --mode "allGT" --save_raw_pred --batch_size 8 --th_cell 0.07 --th_seed 0.35
python ./cell_segmentation.py --inference --cell_type "Fluo-C3DH-H157" --mode "allGT" --save_raw_pred --batch_size 8 --th_cell 0.07 --th_seed 0.35
python ./cell_segmentation.py --inference --cell_type "Fluo-C3DL-MDA231" --mode "allGT" --save_raw_pred --batch_size 8 --th_cell 0.07 --th_seed 0.35
python ./cell_segmentation.py --inference --cell_type "Fluo-N2DH-GOWT1" --mode "allGT" --save_raw_pred --batch_size 16 --th_cell 0.07 --th_seed 0.35
python ./cell_segmentation.py --inference --cell_type "Fluo-N2DL-HeLa" --mode "allGT" --save_raw_pred --batch_size 16 --th_cell 0.07 --th_seed 0.35
python ./cell_segmentation.py --inference --cell_type "Fluo-N3DH-CE" --mode "allGT" --save_raw_pred --batch_size 8 --th_cell 0.07 --th_seed 0.35
python ./cell_segmentation.py --inference --cell_type "Fluo-N3DH-CHO" --mode "allGT" --save_raw_pred --batch_size 8 --th_cell 0.07 --th_seed 0.35
python ./cell_segmentation.py --inference --cell_type "PhC-C2DH-U373" --mode "allGT" --save_raw_pred --batch_size 8 --th_cell 0.07 --th_seed 0.35
python ./cell_segmentation.py --inference --cell_type "PhC-C2DL-PSC" --mode "allGT" --save_raw_pred --batch_size 16 --th_cell 0.07 --th_seed 0.35

python ./cell_segmentation.py --inference --cell_type "BF-C2DL-HSC" --mode "ST" --save_raw_pred --batch_size 16 --th_cell 0.07 --th_seed 0.35 --artifact_correction
python ./cell_segmentation.py --inference --cell_type "BF-C2DL-MuSC" --mode "ST" --save_raw_pred --batch_size 8 --th_cell 0.07 --th_seed 0.35 --artifact_correction
python ./cell_segmentation.py --inference --cell_type "DIC-C2DH-HeLa" --mode "ST" --save_raw_pred --batch_size 16 --th_cell 0.07 --th_seed 0.35
python ./cell_segmentation.py --inference --cell_type "Fluo-C2DL-MSC" --mode "ST" --save_raw_pred --batch_size 16 --th_cell 0.07 --th_seed 0.35
python ./cell_segmentation.py --inference --cell_type "Fluo-C3DH-A549" --mode "ST" --save_raw_pred --batch_size 8 --th_cell 0.07 --th_seed 0.35 --fuse_z_seeds
python ./cell_segmentation.py --inference --cell_type "Fluo-C3DH-H157" --mode "ST" --save_raw_pred --batch_size 8 --th_cell 0.07 --th_seed 0.35 --fuse_z_seeds
python ./cell_segmentation.py --inference --cell_type "Fluo-C3DL-MDA231" --mode "ST" --save_raw_pred --batch_size 8 --th_cell 0.07 --th_seed 0.45 --fuse_z_seeds
python ./cell_segmentation.py --inference --cell_type "Fluo-N2DH-GOWT1" --mode "ST" --save_raw_pred --batch_size 16 --th_cell 0.07 --th_seed 0.45
python ./cell_segmentation.py --inference --cell_type "Fluo-N2DL-HeLa" --mode "ST" --save_raw_pred --batch_size 16 --th_cell 0.07 --th_seed 0.35
python ./cell_segmentation.py --inference --cell_type "Fluo-N3DH-CE" --mode "ST" --save_raw_pred --batch_size 8 --th_cell 0.07 --th_seed 0.45
python ./cell_segmentation.py --inference --cell_type "Fluo-N3DH-CHO" --mode "ST" --save_raw_pred --batch_size 8 --th_cell 0.07 --th_seed 0.45  --fuse_z_seeds
python ./cell_segmentation.py --inference --cell_type "PhC-C2DH-U373" --mode "ST" --save_raw_pred --batch_size 8 --th_cell 0.07 --th_seed 0.35
python ./cell_segmentation.py --inference --cell_type "PhC-C2DL-PSC" --mode "ST" --save_raw_pred --batch_size 16 --th_cell 0.07 --th_seed 0.35

python ./cell_segmentation.py --inference --cell_type "BF-C2DL-HSC" --mode "allST" --save_raw_pred --batch_size 16 --th_cell 0.07 --th_seed 0.35
python ./cell_segmentation.py --inference --cell_type "BF-C2DL-MuSC" --mode "allST" --save_raw_pred --batch_size 8 --th_cell 0.07 --th_seed 0.35
python ./cell_segmentation.py --inference --cell_type "DIC-C2DH-HeLa" --mode "allST" --save_raw_pred --batch_size 16 --th_cell 0.07 --th_seed 0.35
python ./cell_segmentation.py --inference --cell_type "Fluo-C2DL-MSC" --mode "allST" --save_raw_pred --batch_size 16 --th_cell 0.07 --th_seed 0.35
python ./cell_segmentation.py --inference --cell_type "Fluo-C3DH-A549" --mode "allST" --save_raw_pred --batch_size 8 --th_cell 0.07 --th_seed 0.35
python ./cell_segmentation.py --inference --cell_type "Fluo-C3DH-H157" --mode "allST" --save_raw_pred --batch_size 8 --th_cell 0.07 --th_seed 0.35
python ./cell_segmentation.py --inference --cell_type "Fluo-C3DL-MDA231" --mode "allST" --save_raw_pred --batch_size 8 --th_cell 0.07 --th_seed 0.35
python ./cell_segmentation.py --inference --cell_type "Fluo-N2DH-GOWT1" --mode "allST" --save_raw_pred --batch_size 16 --th_cell 0.07 --th_seed 0.35
python ./cell_segmentation.py --inference --cell_type "Fluo-N2DL-HeLa" --mode "allST" --save_raw_pred --batch_size 16 --th_cell 0.07 --th_seed 0.35
python ./cell_segmentation.py --inference --cell_type "Fluo-N3DH-CE" --mode "allST" --save_raw_pred --batch_size 8 --th_cell 0.07 --th_seed 0.35
python ./cell_segmentation.py --inference --cell_type "Fluo-N3DH-CHO" --mode "allST" --save_raw_pred --batch_size 8 --th_cell 0.07 --th_seed 0.35
python ./cell_segmentation.py --inference --cell_type "PhC-C2DH-U373" --mode "allST" --save_raw_pred --batch_size 8 --th_cell 0.07 --th_seed 0.35
python ./cell_segmentation.py --inference --cell_type "PhC-C2DL-PSC" --mode "allST" --save_raw_pred --batch_size 16 --th_cell 0.07 --th_seed 0.35

python ./cell_segmentation.py --inference --cell_type "BF-C2DL-HSC" --mode "GT+ST" --save_raw_pred --batch_size 16 --th_cell 0.07 --th_seed 0.35 --artifact_correction
python ./cell_segmentation.py --inference --cell_type "BF-C2DL-MuSC" --mode "GT+ST" --save_raw_pred --batch_size 8 --th_cell 0.07 --th_seed 0.35 --artifact_correction
python ./cell_segmentation.py --inference --cell_type "DIC-C2DH-HeLa" --mode "GT+ST" --save_raw_pred --batch_size 16 --th_cell 0.07 --th_seed 0.35
python ./cell_segmentation.py --inference --cell_type "Fluo-C2DL-MSC" --mode "GT+ST" --save_raw_pred --batch_size 16 --th_cell 0.07 --th_seed 0.35
python ./cell_segmentation.py --inference --cell_type "Fluo-C3DH-A549" --mode "GT+ST" --save_raw_pred --batch_size 8 --th_cell 0.07 --th_seed 0.35 --fuse_z_seeds
python ./cell_segmentation.py --inference --cell_type "Fluo-C3DH-H157" --mode "GT+ST" --save_raw_pred --batch_size 8 --th_cell 0.07 --th_seed 0.45 --fuse_z_seeds
python ./cell_segmentation.py --inference --cell_type "Fluo-C3DL-MDA231" --mode "GT+ST" --save_raw_pred --batch_size 8 --th_cell 0.07 --th_seed 0.45 --fuse_z_seeds
python ./cell_segmentation.py --inference --cell_type "Fluo-N2DH-GOWT1" --mode "GT+ST" --save_raw_pred --batch_size 16 --th_cell 0.07 --th_seed 0.45
python ./cell_segmentation.py --inference --cell_type "Fluo-N2DL-HeLa" --mode "GT+ST" --save_raw_pred --batch_size 16 --th_cell 0.07 --th_seed 0.35
python ./cell_segmentation.py --inference --cell_type "Fluo-N3DH-CE" --mode "GT+ST" --save_raw_pred --batch_size 8 --th_cell 0.07 --th_seed 0.45
python ./cell_segmentation.py --inference --cell_type "Fluo-N3DH-CHO" --mode "GT+ST" --save_raw_pred --batch_size 8 --th_cell 0.07 --th_seed 0.45  --fuse_z_seeds
python ./cell_segmentation.py --inference --cell_type "PhC-C2DH-U373" --mode "GT+ST" --save_raw_pred --batch_size 8 --th_cell 0.07 --th_seed 0.35
python ./cell_segmentation.py --inference --cell_type "PhC-C2DL-PSC" --mode "GT+ST" --save_raw_pred --batch_size 16 --th_cell 0.07 --th_seed 0.45

python ./cell_segmentation.py --inference --cell_type "BF-C2DL-HSC" --mode "allGT+allST" --save_raw_pred --batch_size 16 --th_cell 0.07 --th_seed 0.45
python ./cell_segmentation.py --inference --cell_type "BF-C2DL-MuSC" --mode "allGT+allST" --save_raw_pred --batch_size 8 --th_cell 0.07 --th_seed 0.45
python ./cell_segmentation.py --inference --cell_type "DIC-C2DH-HeLa" --mode "allGT+allST" --save_raw_pred --batch_size 16 --th_cell 0.07 --th_seed 0.45
python ./cell_segmentation.py --inference --cell_type "Fluo-C2DL-MSC" --mode "allGT+allST" --save_raw_pred --batch_size 16 --th_cell 0.07 --th_seed 0.45
python ./cell_segmentation.py --inference --cell_type "Fluo-C3DH-A549" --mode "allGT+allST" --save_raw_pred --batch_size 8 --th_cell 0.07 --th_seed 0.45
python ./cell_segmentation.py --inference --cell_type "Fluo-C3DH-H157" --mode "allGT+allST" --save_raw_pred --batch_size 8 --th_cell 0.07 --th_seed 0.45
python ./cell_segmentation.py --inference --cell_type "Fluo-C3DL-MDA231" --mode "allGT+allST" --save_raw_pred --batch_size 8 --th_cell 0.07 --th_seed 0.45
python ./cell_segmentation.py --inference --cell_type "Fluo-N2DH-GOWT1" --mode "allGT+allST" --save_raw_pred --batch_size 16 --th_cell 0.07 --th_seed 0.45
python ./cell_segmentation.py --inference --cell_type "Fluo-N2DL-HeLa" --mode "allGT+allST" --save_raw_pred --batch_size 16 --th_cell 0.07 --th_seed 0.45
python ./cell_segmentation.py --inference --cell_type "Fluo-N3DH-CE" --mode "allGT+allST" --save_raw_pred --batch_size 8 --th_cell 0.07 --th_seed 0.45
python ./cell_segmentation.py --inference --cell_type "Fluo-N3DH-CHO" --mode "allGT+allST" --save_raw_pred --batch_size 8 --th_cell 0.07 --th_seed 0.45
python ./cell_segmentation.py --inference --cell_type "PhC-C2DH-U373" --mode "allGT+allST" --save_raw_pred --batch_size 8 --th_cell 0.07 --th_seed 0.45
python ./cell_segmentation.py --inference --cell_type "PhC-C2DL-PSC" --mode "allGT+allST" --save_raw_pred --batch_size 16 --th_cell 0.07 --th_seed 0.45

conda deactivate

