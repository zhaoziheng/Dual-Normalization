import json
import os

nnunet_name_mapping = {
    'CHAOSMR_T1':'Dataset991_CHAOSMR_T1_PNG',
    'CHAOSMR_T2SPIR':'Dataset992_CHAOSMR_T2SPIR_PNG',
    'AMOS22_CT':'Dataset950_AMOS22CT_PNG',
    'MSD_Prostate_ADC':'Dataset994_MSD_Prostate_ADC_PNG',
    'MSD_Prostate_T2':'Dataset995_MSD_Prostate_T2_PNG',
    'PROMISE12_T2':'Dataset990_PROMISE12_T2_PNG',
    'LiQA_T1':'Dataset989_LiQA_GED4_PNG',
    'ATLAS_T1':'Dataset985_ATLAS_T1CE_PNG',
    'MSDLiver_CT':'Dataset949_MSDLiver_PNG'
}

jsonl_mapping = {
    'CHAOSMR_T1':'CHAOS_MRI/CHAOS_MRI_T1',
    'CHAOSMR_T2SPIR':'CHAOS_MRI/CHAOS_MRI_T2_SPIR',
    'AMOS22_CT':'AMOS22CT/AMOS22CT_CT',
    'MSD_Prostate_ADC':'MSD_Prostate/MSD_Prostate_ADC',
    'MSD_Prostate_T2':'MSD_Prostate/MSD_Prostate_T2',
    'PROMISE12_T2':'PROMISE12/PROMISE12_T2',
    'LiQA_T1':"LiQA/LiQA_GED4",
    'ATLAS_T1':'ATLAS/ATLAS_T1CE',
    'MSDLiver_CT':'MSD_Liver/MSD_Liver_CT',
}

label_mapping = {
    'CHAOSMR_T1':'CHAOSMR',
    'CHAOSMR_T2SPIR':'CHAOSMR',
    'AMOS22_CT':'AMOS22_CT',
    'MSD_Prostate_ADC':'MSD_Prostate',
    'MSD_Prostate_T2':'MSD_Prostate',
    'PROMISE12_T2':'Prostate',
    'LiQA_T1':"Liver",
    'ATLAS_T1':'Liver_Tumor',
    'MSDLiver_CT':'Liver_Tumor',
}

num_classes = {
    'CHAOSMR_T1':5,
    'CHAOSMR_T2SPIR':5,
    'AMOS22_CT':15,
    'MSD_Prostate_ADC':3,
    'MSD_Prostate_T2':3,
    'PROMISE12_T2':2,
    'LiQA_T1':2,
    'ATLAS_T1':3,
    'MSDLiver_CT':3,
}

transfer_relation = {
    'CHAOSMR_T1':{
        'CHAOSMR_T1',
        'CHAOSMR_T2SPIR',
        'AMOS22_CT'
    },
    'CHAOSMR_T2SPIR':[
        'CHAOSMR_T2SPIR',
        'CHAOSMR_T1',
        'AMOS22_CT',
        'LiQA_T1',
        'ATLAS_T1'
    ],
    'AMOS22_CT':[
        'AMOS22_CT',
        'CHAOSMR_T2SPIR',
        'CHAOSMR_T1'
    ],
    'MSD_Prostate_ADC':[
        'MSD_Prostate_ADC',
        'MSD_Prostate_T2',
        'PROMISE12_T2'
    ],
    'MSD_Prostate_T2':[
        'MSD_Prostate_T2',
        'MSD_Prostate_ADC',
    ],
    'PROMISE12_T2':[
        'PROMISE12_T2',
        'MSD_Prostate_ADC',
    ],
    'LiQA_T1':[
        'LiQA_T1',
        'CHAOSMR_T2SPIR',
    ],
    'ATLAS_T1':[
        'ATLAS_T1',
        'CHAOSMR_T2SPIR',
        'MSDLiver_CT',
    ],
    'MSDLiver_CT':[
        'MSDLiver_CT',
        'ATLAS_T1'
    ]
}

for source_name, target_name_ls in transfer_relation.items():
    
    jsonl = jsonl_mapping[source_name]
    num = num_classes[source_name]
    
    sh_content = f"""#!/bin/bash
set -e

# process and train

srun --quotatype=auto --partition=medai --ntasks=1 --nodes=1 --gpus-per-task=0 --cpus-per-task=24 --mem-per-cpu=32G \\
python /mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/Dual-Normalization/preprocess_func.py \\
--train_jsonl '/mnt/hwfile/medai/zhaoziheng/SAM/MRDiffusion/trainsets/{jsonl}.jsonl' \\
--test_jsonl '/mnt/hwfile/medai/zhaoziheng/SAM/MRDiffusion/testsets/{jsonl}.jsonl' \\
--target_root '/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/Dual-Normalization/data/{source_name}'

srun --quotatype=auto --partition=medai --ntasks=1 --nodes=1 --gpus-per-task=1 --cpus-per-task=24 --mem-per-cpu=32G -x SH-IDC1-10-140-1-[163],SH-IDC1-10-140-0-[222] \\
python /mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/Dual-Normalization/train_dn_unet.py \\
--train_domain_list_1 'ss' \\
--train_domain_list_2 'sd' \\
--n_classes {num} \\
--batch_size 32 \\
--n_epochs 100 \\
--save_step 20 \\
--lr 0.004 \\
--gpu_ids 0 \\
--result_dir '/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/Dual-Normalization/results/{source_name}' \\
--data_dir '/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/Dual-Normalization/data/{source_name}' """

    for target_name in target_name_ls:
        
        nnunet_name = nnunet_name_mapping[target_name]
        source_label = label_mapping[source_name]
        target_label = label_mapping[target_name]
        
        sh_content += f"""
        
# tranfer to {target_name}
        
srun --quotatype=auto --partition=medai --ntasks=1 --nodes=1 --gpus-per-task=1 --cpus-per-task=24 --mem-per-cpu=32G \\
python /mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/Dual-Normalization/test_dn_unet.py \\
--data_dir '/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/Dual-Normalization/data/{target_name}' \\
--n_classes {num} \\
--test_domain_list '' \\
--model_dir '/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/Dual-Normalization/results/{source_name}/model' \\
--pred_label_dir '/mnt/hwfile/medai/zhaoziheng/SAM/nnUNet_data/nnUNet_raw/{nnunet_name}/labelsPred_from_DualNorm_{source_name}' \\
--input_image_dir '/mnt/hwfile/medai/zhaoziheng/SAM/nnUNet_data/nnUNet_raw/{nnunet_name}/imageTs_from_DualNorm_{source_name}' \\
--batch_size 32 \\
--gpu_ids 0

srun --quotatype=auto --partition=medai --ntasks=1 --nodes=1 --gpus-per-task=0 --cpus-per-task=24 --mem-per-cpu=32G \\
python /mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/nnUNet-Related-Project/nnUNet-as-MRDiffusion-Baseline/evaluate/evaluate_nib.py \\
--target_dataset '{target_label}' \\
--source_dataset '{source_label}' \\
--nnunet_name '{nnunet_name}' \\
--gt_dir 'labelsTs' \\
--seg_dir 'labelsPred_from_DualNorm_{source_name}' \\
--img_dir 'imageTs_from_DualNorm_{source_name}' """

    sh_name = f'/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/Dual-Normalization/sbatch/{source_name}.sh'
    with open(sh_name, 'w') as file:
        file.write(sh_content)
    
    print(f'Write to {source_name}.sh')
        
        
        
