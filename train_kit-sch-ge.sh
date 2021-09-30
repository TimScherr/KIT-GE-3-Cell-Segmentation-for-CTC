eval "$(conda shell.bash hook)"

conda activate kit-sch-ge-2021-cell_segmentation_ve

# Specialized models trained on GTs and STs from the reproduced kit-sch-ge training dataset
python ./train.py --cell_type "BF-C2DL-HSC" --act_fun "relu" --iterations 1 --mode "GT+ST" --optimizer "adam" --split "kit-sch-ge"
python ./train.py --cell_type "BF-C2DL-HSC" --act_fun "mish" --iterations 1 --mode "GT+ST" --optimizer "ranger" --split "kit-sch-ge"

python ./train.py --cell_type "BF-C2DL-MuSC" --act_fun "relu" --iterations 1 --mode "GT+ST" --optimizer "adam" --split "kit-sch-ge"
python ./train.py --cell_type "BF-C2DL-MuSC" --act_fun "mish" --iterations 1 --mode "GT+ST" --optimizer "ranger" --split "kit-sch-ge"

python ./train.py --cell_type "DIC-C2DH-HeLa" --act_fun "relu" --iterations 1 --mode "GT+ST" --optimizer "adam" --split "kit-sch-ge"
python ./train.py --cell_type "DIC-C2DH-HeLa" --act_fun "mish" --iterations 1 --mode "GT+ST" --optimizer "ranger" --split "kit-sch-ge"

python ./train.py --cell_type "Fluo-C2DL-MSC" --act_fun "relu" --iterations 1 --mode "GT+ST" --optimizer "adam" --split "kit-sch-ge"
python ./train.py --cell_type "Fluo-C2DL-MSC" --act_fun "mish" --iterations 1 --mode "GT+ST" --optimizer "ranger" --split "kit-sch-ge"

python ./train.py --cell_type "Fluo-C3DH-A549" --act_fun "relu" --iterations 1 --mode "GT+ST" --optimizer "adam" --split "kit-sch-ge"
python ./train.py --cell_type "Fluo-C3DH-A549" --act_fun "mish" --iterations 1 --mode "GT+ST" --optimizer "ranger" --split "kit-sch-ge"

python ./train.py --cell_type "Fluo-C3DH-H157" --act_fun "relu" --iterations 1 --mode "GT+ST" --optimizer "adam" --split "kit-sch-ge"
python ./train.py --cell_type "Fluo-C3DH-H157" --act_fun "mish" --iterations 1 --mode "GT+ST" --optimizer "ranger" --split "kit-sch-ge"

python ./train.py --cell_type "Fluo-C3DL-MDA231" --act_fun "relu" --iterations 1 --mode "GT+ST" --optimizer "adam" --split "kit-sch-ge"
python ./train.py --cell_type "Fluo-C3DL-MDA231" --act_fun "mish" --iterations 1 --mode "GT+ST" --optimizer "ranger" --split "kit-sch-ge"

python ./train.py --cell_type "Fluo-N2DH-GOWT1" --act_fun "relu" --iterations 1 --mode "GT+ST" --optimizer "adam" --split "kit-sch-ge"
python ./train.py --cell_type "Fluo-N2DH-GOWT1" --act_fun "mish" --iterations 1 --mode "GT+ST" --optimizer "ranger" --split "kit-sch-ge"

python ./train.py --cell_type "Fluo-N3DH-CE" --act_fun "relu" --iterations 1 --mode "GT+ST" --optimizer "adam" --split "kit-sch-ge"
python ./train.py --cell_type "Fluo-N3DH-CE" --act_fun "mish" --iterations 1 --mode "GT+ST" --optimizer "ranger" --split "kit-sch-ge"

python ./train.py --cell_type "Fluo-N2DL-HeLa" --act_fun "relu" --iterations 1 --mode "GT+ST" --optimizer "adam" --split "kit-sch-ge"
python ./train.py --cell_type "Fluo-N2DL-HeLa" --act_fun "mish" --iterations 1 --mode "GT+ST" --optimizer "ranger" --split "kit-sch-ge"

python ./train.py --cell_type "Fluo-N3DH-CHO" --act_fun "relu" --iterations 1 --mode "GT+ST" --optimizer "adam" --split "kit-sch-ge"
python ./train.py --cell_type "Fluo-N3DH-CHO" --act_fun "mish" --iterations 1 --mode "GT+ST" --optimizer "ranger" --split "kit-sch-ge"

python ./train.py --cell_type "PhC-C2DH-U373" --act_fun "relu" --iterations 1 --mode "GT+ST" --optimizer "adam" --split "kit-sch-ge"
python ./train.py --cell_type "PhC-C2DH-U373" --act_fun "mish" --iterations 1 --mode "GT+ST" --optimizer "ranger" --split "kit-sch-ge"

python ./train.py --cell_type "PhC-C2DL-PSC" --act_fun "relu" --iterations 1 --mode "GT+ST" --optimizer "adam" --split "kit-sch-ge"
python ./train.py --cell_type "PhC-C2DL-PSC" --act_fun "mish" --iterations 1 --mode "GT+ST" --optimizer "ranger" --split "kit-sch-ge"

conda deactivate

