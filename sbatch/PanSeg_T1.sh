#!/bin/bash
set -e

# process and train

srun --quotatype=auto --partition=medai --ntasks=1 --nodes=1 --gpus-per-task=0 --cpus-per-task=24 --mem-per-cpu=32G \
python /mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/Dual-Normalization/preprocess_func_PanSeg.py \
--modality 'T1'

# srun --quotatype=auto --partition=medai --ntasks=1 --nodes=1 --gpus-per-task=1 --cpus-per-task=24 --mem-per-cpu=32G -x SH-IDC1-10-140-1-[163,157],SH-IDC1-10-140-0-[222,175] \
# python /mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/Dual-Normalization/train_dn_unet.py \
# --train_domain_list_1 'ss' \
# --train_domain_list_2 'sd' \
# --n_classes 2 \
# --batch_size 32 \
# --n_epochs 100 \
# --save_step 10 \
# --lr 0.004 \
# --gpu_ids 0 \
# --result_dir '/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/Dual-Normalization/results/PanSeg_T1' \
# --data_dir '/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/Dual-Normalization/data/PanSeg_T1'

# # in domain

# srun --quotatype=auto --partition=medai --ntasks=1 --nodes=1 --gpus-per-task=1 --cpus-per-task=24 --mem-per-cpu=32G -x SH-IDC1-10-140-1-[163,157],SH-IDC1-10-140-0-[222,175] \
# python /mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/Dual-Normalization/test_dn_unet.py \
# --data_dir '/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/Dual-Normalization/data/PanSeg_T1' \
# --n_classes 2 \
# --test_domain_list '' \
# --model_dir '/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/Dual-Normalization/results/PanSeg_T1/model' \
# --pred_label_dir '/mnt/hwfile/medai/zhaoziheng/SAM/nnUNet_data/nnUNet_raw/Dataset958_PanSeg_T1_PNG/labelsPred_from_DualNorm_PanSeg_T1' \
# --batch_size 32 \
# --gpu_ids 0

# srun --quotatype=auto --partition=medai --ntasks=1 --nodes=1 --gpus-per-task=0 --cpus-per-task=24 --mem-per-cpu=32G \
# python /mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/nnUNet-Related-Project/nnUNet-as-MRDiffusion-Baseline/evaluate/evaluate_nib.py \
# --target_dataset 'Pancreas' \
# --source_dataset 'Pancreas' \
# --nnunet_name 'Dataset958_PanSeg_T1_PNG' \
# --gt_dir 'labelsTs' \
# --seg_dir 'labelsPred_from_DualNorm_PanSeg_T1' \
# --img_dir 'imagesTs' 

# # transfer to PanSeg T2

# srun --quotatype=auto --partition=medai --ntasks=1 --nodes=1 --gpus-per-task=1 --cpus-per-task=24 --mem-per-cpu=32G -x SH-IDC1-10-140-1-[163,157],SH-IDC1-10-140-0-[222,175] \
# python /mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/Dual-Normalization/test_dn_unet.py \
# --data_dir '/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/Dual-Normalization/data/PanSeg_T2' \
# --n_classes 2 \
# --test_domain_list '' \
# --model_dir '/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/Dual-Normalization/results/PanSeg_T1/model' \
# --pred_label_dir '/mnt/hwfile/medai/zhaoziheng/SAM/nnUNet_data/nnUNet_raw/Dataset957_PanSeg_T2_PNG/labelsPred_from_DualNorm_PanSeg_T1' \
# --batch_size 32 \
# --gpu_ids 0

# srun --quotatype=auto --partition=medai --ntasks=1 --nodes=1 --gpus-per-task=0 --cpus-per-task=24 --mem-per-cpu=32G \
# python /mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/nnUNet-Related-Project/nnUNet-as-MRDiffusion-Baseline/evaluate/evaluate_nib.py \
# --target_dataset 'Pancreas' \
# --source_dataset 'Pancreas' \
# --nnunet_name 'Dataset957_PanSeg_T2_PNG' \
# --gt_dir 'labelsTs' \
# --seg_dir 'labelsPred_from_DualNorm_PanSeg_T1' \
# --img_dir 'imagesTs' 