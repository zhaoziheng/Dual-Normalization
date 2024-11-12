#!/bin/bash
set -e

# process and train

srun --quotatype=auto --partition=medai --ntasks=1 --nodes=1 --gpus-per-task=0 --cpus-per-task=24 --mem-per-cpu=32G \
python /mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/Dual-Normalization/preprocess_func.py \
--train_jsonl '/mnt/hwfile/medai/zhaoziheng/SAM/MRDiffusion/trainsets/LiQA/LiQA_GED4.jsonl' \
--test_jsonl '/mnt/hwfile/medai/zhaoziheng/SAM/MRDiffusion/testsets/LiQA/LiQA_GED4.jsonl' \
--target_root '/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/Dual-Normalization/data/LiQA_T1'

# srun --quotatype=auto --partition=medai --ntasks=1 --nodes=1 --gpus-per-task=1 --cpus-per-task=24 --mem-per-cpu=32G -x SH-IDC1-10-140-1-[163,157],SH-IDC1-10-140-0-[222] \
# python /mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/Dual-Normalization/train_dn_unet.py \
# --train_domain_list_1 'ss' \
# --train_domain_list_2 'sd' \
# --n_classes 2 \
# --batch_size 32 \
# --n_epochs 100 \
# --save_step 20 \
# --lr 0.004 \
# --gpu_ids 0 \
# --result_dir '/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/Dual-Normalization/results/LiQA_T1' \
# --data_dir '/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/Dual-Normalization/data/LiQA_T1' 
        
# # tranfer to LiQA_T1
        
# srun --quotatype=auto --partition=medai --ntasks=1 --nodes=1 --gpus-per-task=1 --cpus-per-task=24 --mem-per-cpu=32G -x SH-IDC1-10-140-1-[163,157],SH-IDC1-10-140-0-[222] \
# python /mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/Dual-Normalization/test_dn_unet.py \
# --data_dir '/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/Dual-Normalization/data/LiQA_T1' \
# --n_classes 2 \
# --test_domain_list '' \
# --model_dir '/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/Dual-Normalization/results/LiQA_T1/model' \
# --pred_label_dir '/mnt/hwfile/medai/zhaoziheng/SAM/nnUNet_data/nnUNet_raw/Dataset989_LiQA_GED4_PNG/labelsPred_from_DualNorm_LiQA_T1' \
# --input_image_dir '/mnt/hwfile/medai/zhaoziheng/SAM/nnUNet_data/nnUNet_raw/Dataset989_LiQA_GED4_PNG/imageTs_from_DualNorm_LiQA_T1' \
# --batch_size 32 \
# --gpu_ids 0

# srun --quotatype=auto --partition=medai --ntasks=1 --nodes=1 --gpus-per-task=0 --cpus-per-task=24 --mem-per-cpu=32G \
# python /mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/nnUNet-Related-Project/nnUNet-as-MRDiffusion-Baseline/evaluate/evaluate_nib.py \
# --target_dataset 'Liver' \
# --source_dataset 'Liver' \
# --nnunet_name 'Dataset989_LiQA_GED4_PNG' \
# --gt_dir 'labelsTs' \
# --seg_dir 'labelsPred_from_DualNorm_LiQA_T1' \
# --img_dir 'imageTs_from_DualNorm_LiQA_T1' 
        
# # tranfer to CHAOSMR_T2SPIR
        
# srun --quotatype=auto --partition=medai --ntasks=1 --nodes=1 --gpus-per-task=1 --cpus-per-task=24 --mem-per-cpu=32G -x SH-IDC1-10-140-1-[163,157],SH-IDC1-10-140-0-[222] \
# python /mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/Dual-Normalization/test_dn_unet.py \
# --data_dir '/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/Dual-Normalization/data/CHAOSMR_T2SPIR' \
# --n_classes 2 \
# --test_domain_list '' \
# --model_dir '/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/Dual-Normalization/results/LiQA_T1/model' \
# --pred_label_dir '/mnt/hwfile/medai/zhaoziheng/SAM/nnUNet_data/nnUNet_raw/Dataset992_CHAOSMR_T2SPIR_PNG/labelsPred_from_DualNorm_LiQA_T1' \
# --input_image_dir '/mnt/hwfile/medai/zhaoziheng/SAM/nnUNet_data/nnUNet_raw/Dataset992_CHAOSMR_T2SPIR_PNG/imageTs_from_DualNorm_LiQA_T1' \
# --batch_size 32 \
# --gpu_ids 0

# srun --quotatype=auto --partition=medai --ntasks=1 --nodes=1 --gpus-per-task=0 --cpus-per-task=24 --mem-per-cpu=32G \
# -x SH-IDC1-10-140-0-[163,157] \
# python /mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/nnUNet-Related-Project/nnUNet-as-MRDiffusion-Baseline/evaluate/evaluate_nib.py \
# --target_dataset 'CHAOSMR' \
# --source_dataset 'Liver' \
# --nnunet_name 'Dataset992_CHAOSMR_T2SPIR_PNG' \
# --gt_dir 'labelsTs' \
# --seg_dir 'labelsPred_from_DualNorm_LiQA_T1' \
# --img_dir 'imageTs_from_DualNorm_LiQA_T1' 