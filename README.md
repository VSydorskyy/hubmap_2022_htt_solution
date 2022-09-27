# hubmap_2022_htt_solution
Codebase for HuBMAP + HPA - Hacking the Human Body: Human Torus Team solution (3d Place) 

# Setting up environment  

1. conda env create -f environment.yml
2. pip install -e . --no-deps

# Prepare Data and CV split

1. cd data
2. kaggle competitions download -c hubmap-organ-segmentation
3. unzip hubmap-organ-segmentation.zip
4. cd ../
5. python scripts/create_cv_split.py data/train.csv --save_path data/cv_split5_v2.npy
6. Download HPA additional data to `data/hpa` folder and unzip all 
    - https://www.kaggle.com/datasets/igorkrashenyi/liver-hpa-pt0
    - https://www.kaggle.com/datasets/igorkrashenyi/liver-hpa-pt2
    - https://www.kaggle.com/datasets/igorkrashenyi/liver-hpa-pt1
    - https://www.kaggle.com/datasets/igorkrashenyi/hap-kidney-dataset-pt1
    - https://www.kaggle.com/datasets/igorkrashenyi/kidney-hpa-dataset-pt0
    - https://www.kaggle.com/datasets/igorkrashenyi/hpa-colon-dataset
    - https://www.kaggle.com/datasets/igorkrashenyi/hpa-spleen-dataset-pt1
    - https://www.kaggle.com/datasets/igorkrashenyi/hpa-spleen-dataset-pt0
    - https://www.kaggle.com/datasets/igorkrashenyi/hpa-prostate-dataset
    - https://www.kaggle.com/datasets/igorkrashenyi/lung-hpa-dataset
7. Download all GTEX additional data in `data/gtex`
    - https://www.kaggle.com/datasets/sakvaua/gtex-pseudo-humantorusteam
# Train First models without Pseudo 

1. CUDA_VISIBLE_DEVICES="{gpu_num}" python scripts/main_train.py train_configs/unet_no_pseudo.py
2. CUDA_VISIBLE_DEVICES="{gpu_num}" python scripts/main_train.py train_configs/unetpp_no_pseudo.py

# Perform Inference (Not obligatory)

1. Find out experiment names in `logdirs`
2. Use  `notebooks/inference_notebook.ipynb` path `EXP_NAME` and `CONFIG` from `unet_no_pseudo.py` (or `unetpp_no_pseudo.py`)
3. Run notebook and get OOF metrics and other evaluation results 

# Create Pseudo Labels (Not obligatory)

1. Use `notebooks/create_pseudo_gtex.ipynb` and `notebooks/create_pseudo.ipynb` for creating pseudo labels for each organ you have to re-run notebook, changing organ path and name in the config. create_pseudo_gtex for GTEX and create_pseudo for HPA

# Download Pseudo Labels

1. You can download pseudo labels directly from Kaggle - https://www.kaggle.com/datasets/vladimirsydor/hubmap-2022-add-data-labels-v2 . It contains them both in .csv (rle) and .png (soft) formats

# Create final Pseudo datasets 

## HPA 

1. Use `notebooks/prepare_pseudo_train_data.ipynb` to aggregate all HPA pseudo in one dataframe and folder. V2 - refers to first pseudo iteration (K=1) and V3 (V3 Full) refers to second pseudo iteration (K=2)

## GTEX

1. Use `notebooks/prepare_pseudo_train_data.ipynb` to aggregate all HPA pseudo in one dataframe and folder. Simply change `data/hpa` to `data/gtex`

# Train Final models with Pseudo

1. CUDA_VISIBLE_DEVICES="{gpu_num}" python scripts/main_train.py train_configs/unet.py
2. CUDA_VISIBLE_DEVICES="{gpu_num}" python scripts/main_train.py train_configs/unetmitb3.py
3. CUDA_VISIBLE_DEVICES="{gpu_num}" python scripts/main_train.py train_configs/unetmitb5.py
4. CUDA_VISIBLE_DEVICES="{gpu_num}" python scripts/main_train.py train_configs/unetpp.py

# Evaluate ensemble  

1. Use `notebooks/inference_notebook_ensem.ipynb` for evaluating ensemble just define `EXP_NAME`, `CONFIG` and `ADITIONAL_MODELS` from trained experiments from previous steps (from `logdirs`)