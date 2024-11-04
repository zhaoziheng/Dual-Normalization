#!/bin/bash
set -e

# process and train

# srun --quotatype=auto --partition=medai --ntasks=1 --nodes=1 --gpus-per-task=0 --cpus-per-task=24 --mem-per-cpu=32G \
# # python /mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/Dual-Normalization/preprocess_func.py \
# --train_jsonl '/mnt/hwfile/medai/zhaoziheng/SAM/MRDiffusion/trainsets/MSD_Prostate/MSD_Prostate_T2.jsonl' \
# --test_jsonl '/mnt/hwfile/medai/zhaoziheng/SAM/MRDiffusion/testsets/MSD_Prostate/MSD_Prostate_T2.jsonl' \
# --target_root '/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/Dual-Normalization/data/MSD_Prostate_T2'

# srun --quotatype=auto --partition=medai --ntasks=1 --nodes=1 --gpus-per-task=1 --cpus-per-task=24 --mem-per-cpu=32G -x SH-IDC1-10-140-1-[163],SH-IDC1-10-140-0-[222] \
# # python /mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/Dual-Normalization/train_dn_unet.py \
# --train_domain_list_1 'ss' \
# --train_domain_list_2 'sd' \
# --n_classes 3 \
# --batch_size 32 \
# --n_epochs 100 \
# --save_step 20 \
# --lr 0.004 \
# --gpu_ids 0 \
# --result_dir '/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/Dual-Normalization/results/MSD_Prostate_T2' \
# --data_dir '/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/Dual-Normalization/data/MSD_Prostate_T2' 
        
# tranfer to MSD_Prostate_T2
        
srun --quotatype=auto --partition=medai --ntasks=1 --nodes=1 --gpus-per-task=1 --cpus-per-task=24 --mem-per-cpu=32G -x SH-IDC1-10-140-1-[163],SH-IDC1-10-140-0-[222] \
python /mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/Dual-Normalization/test_dn_unet.py \
--data_dir '/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/Dual-Normalization/data/MSD_Prostate_T2' \
--n_classes 3 \
--test_domain_list '' \
--model_dir '/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/Dual-Normalization/results/MSD_Prostate_T2/model' \
--pred_label_dir '/mnt/hwfile/medai/zhaoziheng/SAM/nnUNet_data/nnUNet_raw/Dataset995_MSD_Prostate_T2_PNG/labelsPred_from_DualNorm_MSD_Prostate_T2' \
--input_image_dir '/mnt/hwfile/medai/zhaoziheng/SAM/nnUNet_data/nnUNet_raw/Dataset995_MSD_Prostate_T2_PNG/imageTs_from_DualNorm_MSD_Prostate_T2' \
--batch_size 32 \
--gpu_ids 0

srun --quotatype=auto --partition=medai --ntasks=1 --nodes=1 --gpus-per-task=0 --cpus-per-task=24 --mem-per-cpu=32G \
python /mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/nnUNet-Related-Project/nnUNet-as-MRDiffusion-Baseline/evaluate/evaluate_nib.py \
--target_dataset 'MSD_Prostate' \
--source_dataset 'MSD_Prostate' \
--nnunet_name 'Dataset995_MSD_Prostate_T2_PNG' \
--gt_dir 'labelsTs' \
--seg_dir 'labelsPred_from_DualNorm_MSD_Prostate_T2' \
--img_dir 'imageTs_from_DualNorm_MSD_Prostate_T2' 
        
# tranfer to MSD_Prostate_ADC
        
srun --quotatype=auto --partition=medai --ntasks=1 --nodes=1 --gpus-per-task=1 --cpus-per-task=24 --mem-per-cpu=32G -x SH-IDC1-10-140-1-[163],SH-IDC1-10-140-0-[222] \
python /mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/Dual-Normalization/test_dn_unet.py \
--data_dir '/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/Dual-Normalization/data/MSD_Prostate_ADC' \
--n_classes 3 \
--test_domain_list '' \
--model_dir '/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/Dual-Normalization/results/MSD_Prostate_T2/model' \
--pred_label_dir '/mnt/hwfile/medai/zhaoziheng/SAM/nnUNet_data/nnUNet_raw/Dataset994_MSD_Prostate_ADC_PNG/labelsPred_from_DualNorm_MSD_Prostate_T2' \
--input_image_dir '/mnt/hwfile/medai/zhaoziheng/SAM/nnUNet_data/nnUNet_raw/Dataset994_MSD_Prostate_ADC_PNG/imageTs_from_DualNorm_MSD_Prostate_T2' \
--batch_size 32 \
--gpu_ids 0

srun --quotatype=auto --partition=medai --ntasks=1 --nodes=1 --gpus-per-task=0 --cpus-per-task=24 --mem-per-cpu=32G \
python /mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/nnUNet-Related-Project/nnUNet-as-MRDiffusion-Baseline/evaluate/evaluate_nib.py \
--target_dataset 'MSD_Prostate' \
--source_dataset 'MSD_Prostate' \
--nnunet_name 'Dataset994_MSD_Prostate_ADC_PNG' \
--gt_dir 'labelsTs' \
--seg_dir 'labelsPred_from_DualNorm_MSD_Prostate_T2' \
--img_dir 'imageTs_from_DualNorm_MSD_Prostate_T2' 