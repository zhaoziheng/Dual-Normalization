import os
import shutil
from pathlib import Path

for dataset in ['AMOS22_CT', 'ATLAS_T1', 'CHAOSMR_T1', 'CHAOSMR_T2SPIR', 'LiQA_T1', 'MSD_Prostate_ADC', 'MSD_Prostate_T2', 'MSDLiver_CT', 'PanSeg_T1', 'PanSeg_T2', 'PROMISE12_T2']:

        # source_directory = os.path.join('/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/Dual-Normalization/data', dataset, 'visual_ss_sd')
        # target_directory = os.path.join('/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/Dual-Normalization/data', dataset, 'visual_all_sd')

        # if not os.path.exists(target_directory):
        #     os.makedirs(target_directory)

        # for filename in os.listdir(source_directory):
        #     if 'nonlinear' in filename and filename.endswith('.png'):
        #         shutil.move(os.path.join(source_directory, filename), target_directory)
                
        # source_directory = os.path.join('/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/Dual-Normalization/data', dataset, 'visual_test_sd')
        # target_directory = os.path.join('/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/Dual-Normalization/data', dataset, 'visual_all_sd')

        # if not os.path.exists(target_directory):
        #     os.makedirs(target_directory)

        # for filename in os.listdir(source_directory):
        #     if 'nonlinear' in filename and filename.endswith('.png'):
        #         shutil.move(os.path.join(source_directory, filename), target_directory)
        
        a_directory = os.path.join('/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/Dual-Normalization/data', dataset, 'visual_all_sd')
        b_directory = os.path.join('/mnt/hwfile/medai/zhaoziheng/SAM/Dual-Normalization/', dataset, 'visual_all_sd')
        
        Path.mkdir(Path(b_directory), parents=True, exist_ok=True)
            
        shutil.copytree(a_directory, b_directory, dirs_exist_ok=True)