eval "$(conda shell.bash hook)"

conda activate kit-sch-ge-2021-cell_segmentation_ve

python ./cell_segmentation.py --train --cell_type "BF-C2DL-HSC" --mode "GT"
python ./cell_segmentation.py --evaluate --cell_type "BF-C2DL-HSC" --mode "GT" --save_raw_pred --batch_size 16 --artifact_correction
python ./cell_segmentation.py --train --cell_type "BF-C2DL-MuSC" --mode "GT"
python ./cell_segmentation.py --evaluate --cell_type "BF-C2DL-MuSC" --mode "GT" --save_raw_pred --batch_size 8 --artifact_correction
python ./cell_segmentation.py --train --cell_type "DIC-C2DH-HeLa" --mode "GT"
python ./cell_segmentation.py --evaluate --cell_type "DIC-C2DH-HeLa" --mode "GT" --save_raw_pred --batch_size 16
python ./cell_segmentation.py --train --cell_type "Fluo-C2DL-MSC" --mode "GT"
python ./cell_segmentation.py --evaluate --cell_type "Fluo-C2DL-MSC" --mode "GT" --save_raw_pred --batch_size 16
python ./cell_segmentation.py --train --cell_type "Fluo-C3DH-A549" --mode "GT"
python ./cell_segmentation.py --evaluate --cell_type "Fluo-C3DH-A549" --mode "GT" --save_raw_pred --batch_size 8
python ./cell_segmentation.py --train --cell_type "Fluo-C3DH-H157" --mode "GT"
python ./cell_segmentation.py --evaluate --cell_type "Fluo-C3DH-H157" --mode "GT" --save_raw_pred --batch_size 8
python ./cell_segmentation.py --train --cell_type "Fluo-C3DL-MDA231" --mode "GT"
python ./cell_segmentation.py --evaluate --cell_type "Fluo-C3DL-MDA231" --mode "GT" --save_raw_pred --batch_size 8
python ./cell_segmentation.py --train --cell_type "Fluo-N2DH-GOWT1" --mode "GT"
python ./cell_segmentation.py --evaluate --cell_type "Fluo-N2DH-GOWT1" --mode "GT" --save_raw_pred --batch_size 16
python ./cell_segmentation.py --train --cell_type "Fluo-N3DH-CE" --mode "GT"
python ./cell_segmentation.py --evaluate --cell_type "Fluo-N3DH-CE" --mode "GT" --save_raw_pred --batch_size 8
python ./cell_segmentation.py --train --cell_type "Fluo-N2DL-HeLa" --mode "GT"
python ./cell_segmentation.py --evaluate --cell_type "Fluo-N2DL-HeLa" --mode "GT" --save_raw_pred --batch_size 16
python ./cell_segmentation.py --train --cell_type "Fluo-N3DH-CHO" --mode "GT"
python ./cell_segmentation.py --evaluate --cell_type "Fluo-N3DH-CHO" --mode "GT" --save_raw_pred --batch_size 8
python ./cell_segmentation.py --train --cell_type "PhC-C2DH-U373" --mode "GT"
python ./cell_segmentation.py --evaluate --cell_type "PhC-C2DH-U373" --mode "GT" --save_raw_pred --batch_size 16
python ./cell_segmentation.py --train --cell_type "PhC-C2DL-PSC" --mode "GT"
python ./cell_segmentation.py --evaluate --cell_type "PhC-C2DL-PSC" --mode "GT" --save_raw_pred --batch_size 16

python ./cell_segmentation.py --train --cell_type "all" --mode "allGT"
python ./cell_segmentation.py --evaluate --cell_type "all" --mode "allGT" --save_raw_pred --batch_size 8

python ./cell_segmentation.py --train --cell_type "BF-C2DL-HSC" --mode "ST"
python ./cell_segmentation.py --evaluate --cell_type "BF-C2DL-HSC" --mode "ST" --save_raw_pred --batch_size 16 --artifact_correction
python ./cell_segmentation.py --train --cell_type "BF-C2DL-MuSC" --mode "ST"
python ./cell_segmentation.py --evaluate --cell_type "BF-C2DL-MuSC" --mode "ST" --save_raw_pred --batch_size 8 --artifact_correction
python ./cell_segmentation.py --train --cell_type "DIC-C2DH-HeLa" --mode "ST"
python ./cell_segmentation.py --evaluate --cell_type "DIC-C2DH-HeLa" --mode "ST" --save_raw_pred --batch_size 16
python ./cell_segmentation.py --train --cell_type "Fluo-C2DL-MSC" --mode "ST"
python ./cell_segmentation.py --evaluate --cell_type "Fluo-C2DL-MSC" --mode "ST" --save_raw_pred --batch_size 16
python ./cell_segmentation.py --train --cell_type "Fluo-C3DH-A549" --mode "ST"
python ./cell_segmentation.py --evaluate --cell_type "Fluo-C3DH-A549" --mode "ST" --save_raw_pred --batch_size 8
python ./cell_segmentation.py --train --cell_type "Fluo-C3DH-H157" --mode "ST"
python ./cell_segmentation.py --evaluate --cell_type "Fluo-C3DH-H157" --mode "ST" --save_raw_pred --batch_size 8
python ./cell_segmentation.py --train --cell_type "Fluo-C3DL-MDA231" --mode "ST"
python ./cell_segmentation.py --evaluate --cell_type "Fluo-C3DL-MDA231" --mode "ST" --save_raw_pred --batch_size 8
python ./cell_segmentation.py --train --cell_type "Fluo-N2DH-GOWT1" --mode "ST"
python ./cell_segmentation.py --evaluate --cell_type "Fluo-N2DH-GOWT1" --mode "ST" --save_raw_pred --batch_size 16
python ./cell_segmentation.py --train --cell_type "Fluo-N3DH-CE" --mode "ST"
python ./cell_segmentation.py --evaluate --cell_type "Fluo-N3DH-CE" --mode "ST" --save_raw_pred --batch_size 8
python ./cell_segmentation.py --train --cell_type "Fluo-N2DL-HeLa" --mode "ST"
python ./cell_segmentation.py --evaluate --cell_type "Fluo-N2DL-HeLa" --mode "ST" --save_raw_pred --batch_size 16
python ./cell_segmentation.py --train --cell_type "Fluo-N3DH-CHO" --mode "ST"
python ./cell_segmentation.py --evaluate --cell_type "Fluo-N3DH-CHO" --mode "ST" --save_raw_pred --batch_size 8
python ./cell_segmentation.py --train --cell_type "PhC-C2DH-U373" --mode "ST"
python ./cell_segmentation.py --evaluate --cell_type "PhC-C2DH-U373" --mode "ST" --save_raw_pred --batch_size 16
python ./cell_segmentation.py --train --cell_type "PhC-C2DL-PSC" --mode "ST"
python ./cell_segmentation.py --evaluate --cell_type "PhC-C2DL-PSC" --mode "ST" --save_raw_pred --batch_size 16

python ./cell_segmentation.py --train --cell_type "all" --mode "allST"
python ./cell_segmentation.py --evaluate --cell_type "all" --mode "allST" --save_raw_pred --batch_size 8

python ./cell_segmentation.py --train --cell_type "BF-C2DL-HSC" --mode "GT+ST"
python ./cell_segmentation.py --evaluate --cell_type "BF-C2DL-HSC" --mode "GT+ST" --save_raw_pred --batch_size 16 --artifact_correction
python ./cell_segmentation.py --train --cell_type "BF-C2DL-MuSC" --mode "GT+ST"
python ./cell_segmentation.py --evaluate --cell_type "BF-C2DL-MuSC" --mode "GT+ST" --save_raw_pred --batch_size 8 --artifact_correction
python ./cell_segmentation.py --train --cell_type "DIC-C2DH-HeLa" --mode "GT+ST"
python ./cell_segmentation.py --evaluate --cell_type "DIC-C2DH-HeLa" --mode "GT+ST" --save_raw_pred --batch_size 16
python ./cell_segmentation.py --train --cell_type "Fluo-C2DL-MSC" --mode "GT+ST"
python ./cell_segmentation.py --evaluate --cell_type "Fluo-C2DL-MSC" --mode "GT+ST" --save_raw_pred --batch_size 16
python ./cell_segmentation.py --train --cell_type "Fluo-C3DH-A549" --mode "GT+ST"
python ./cell_segmentation.py --evaluate --cell_type "Fluo-C3DH-A549" --mode "GT+ST" --save_raw_pred --batch_size 8
python ./cell_segmentation.py --train --cell_type "Fluo-C3DH-H157" --mode "GT+ST"
python ./cell_segmentation.py --evaluate --cell_type "Fluo-C3DH-H157" --mode "GT+ST" --save_raw_pred --batch_size 8
python ./cell_segmentation.py --train --cell_type "Fluo-C3DL-MDA231" --mode "GT+ST"
python ./cell_segmentation.py --evaluate --cell_type "Fluo-C3DL-MDA231" --mode "GT+ST" --save_raw_pred --batch_size 8
python ./cell_segmentation.py --train --cell_type "Fluo-N2DH-GOWT1" --mode "GT+ST"
python ./cell_segmentation.py --evaluate --cell_type "Fluo-N2DH-GOWT1" --mode "GT+ST" --save_raw_pred --batch_size 16
python ./cell_segmentation.py --train --cell_type "Fluo-N3DH-CE" --mode "GT+ST"
python ./cell_segmentation.py --evaluate --cell_type "Fluo-N3DH-CE" --mode "GT+ST" --save_raw_pred --batch_size 8
python ./cell_segmentation.py --train --cell_type "Fluo-N2DL-HeLa" --mode "GT+ST"
python ./cell_segmentation.py --evaluate --cell_type "Fluo-N2DL-HeLa" --mode "GT+ST" --save_raw_pred --batch_size 16
python ./cell_segmentation.py --train --cell_type "Fluo-N3DH-CHO" --mode "GT+ST"
python ./cell_segmentation.py --evaluate --cell_type "Fluo-N3DH-CHO" --mode "GT+ST" --save_raw_pred --batch_size 8
python ./cell_segmentation.py --train --cell_type "PhC-C2DH-U373" --mode "GT+ST"
python ./cell_segmentation.py --evaluate --cell_type "PhC-C2DH-U373" --mode "GT+ST" --save_raw_pred --batch_size 16
python ./cell_segmentation.py --train --cell_type "PhC-C2DL-PSC" --mode "GT+ST"
python ./cell_segmentation.py --evaluate --cell_type "PhC-C2DL-PSC" --mode "GT+ST" --save_raw_pred --batch_size 16

python ./cell_segmentation.py --train --cell_type "all" --mode "allGT+allST"
python ./cell_segmentation.py --evaluate --cell_type "all" --mode "allGT+allST" --save_raw_pred --batch_size 8

conda deactivate

