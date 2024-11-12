import nibabel as nib
import numpy as np
import os
import SimpleITK as sitk
from bezier_curve import bezier_curve
from tqdm import tqdm
import json
import torch
from torchvision import transforms
import monai
from einops import rearrange, repeat, reduce
import cv2
import random
import shutil

import matplotlib.pyplot as plt

class Loader_Wrapper():
    """
    different from SAT format, when trans to nnUNET 2D format (png):
    1. no spacing and no crop
    2. no merged labels and masks
    3. no intensity normalization
    """
    
    def __init__(self):
        self.to_tensor = transforms.ToTensor()

    def CHAOS_MRI(self, datum:dict) -> tuple:
        """
        'liver', 
        'right kidney', 
        'left kidney', 
        'spleen'
        """
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.EnsureChannelFirstd(keys=['image']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                # monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                # monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image']
        mask = dictionary['label']
        
        labels = datum['label'][:4]
        
        # NOTE: merge label
        # kidney = mask[1] + mask[2]
        # mask = torch.cat((mask, kidney.unsqueeze(0)), dim=0)
        # labels.append("kidney")
        
        #img = repeat(img, 'c h w d -> (c r) h w d', r=3)
        mask = (mask-mask.min())/(mask.max()-mask.min() + 1e-10) # normal到0~1之间
        mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        # img = Normalization(img, 'MRI')
        
        return img, mask, labels, datum['modality'], datum['image'], datum['mask']
    
    def PROMISE12(self, datum:dict) -> tuple:
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.EnsureChannelFirstd(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                # monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                # monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image']
        mask = dictionary['label']
        
        labels = datum['label']
        
        #img = repeat(img, 'c h w d -> (c r) h w d', r=3)
        mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        # img = Normalization(img, 'MRI')
        
        return img, mask, labels, datum['modality'], datum['image'], datum['mask']
    
    def AMOS22_CT(self, datum:dict) -> tuple:
        """
        labels = [
            'spleen', 
            'right kidney',
            'left kidney',
            'gallbladder',
            'esophagus',
            'liver',
            'stomach',
            'aorta',
            'inferior vena cava',
            'pancreas',
            'right adrenal gland',
            'left adrenal gland',
            'duodenum',
            'urinary bladder',
            ]
            
        """
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.EnsureChannelFirstd(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                # monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                # monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image'] # [1, H, W, D]
        mask = dictionary['label']# [1, H, W, D]
        
        labels = datum['label'][:14]
        mc_masks = []
        for i, label in enumerate(labels):
            binary_mask = torch.where(mask==(i+1), 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)
        
        # merge label
        # mc_masks.append(mc_masks[1]+mc_masks[2])
        # labels.append("kidney")
        # mc_masks.append(mc_masks[10]+mc_masks[11])
        # labels.append("adrenal gland")
        
        mask = torch.concat(mc_masks, dim=0) # [C, H, W, D]
        
        ##img = repeat(img, 'c h w d -> (c r) h w d', r=3)
        mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        # img = Normalization(img, 'CT')

        return img, mask, labels, datum['modality'], datum['image'], datum['mask']
    
    def AMOS22_MRI(self, datum:dict) -> tuple:
        """
        labels = [
            'spleen', 
            'right kidney',
            'left kidney',
            'gallbladder',
            'esophagus',
            'liver',
            'stomach',
            'aorta',
            'inferior vena cava',
            'pancreas',
            'right adrenal gland',
            'left adrenal gland',
            'duodenum',
            'urinary bladder',
            ]
            
        """
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.EnsureChannelFirstd(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                # monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                # monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image'] # [1, H, W, D]
        mask = dictionary['label']# [1, H, W, D]
        
        labels = datum['label'][:14]
        mc_masks = []
        for i, label in enumerate(labels):
            binary_mask = torch.where(mask==(i+1), 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)
        
        # merge label
        # mc_masks.append(mc_masks[1]+mc_masks[2])
        # labels.append("kidney")
        # mc_masks.append(mc_masks[10]+mc_masks[11])
        # labels.append("adrenal gland")
        
        mask = torch.concat(mc_masks, dim=0) # [C, H, W, D]
        
        ##img = repeat(img, 'c h w d -> (c r) h w d', r=3)
        mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        # img = Normalization(img, 'MRI')

        return img, mask, labels, datum['modality'], datum['image'], datum['mask']
    
    def ATLAS(self, datum:dict) -> tuple:
        """
        labels = [
            "liver",
            "liver tumor",
            ]
            
        """
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.EnsureChannelFirstd(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                # monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                # monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image'] # [1, H, W, D]
        mask = dictionary['label']# [1, H, W, D 
        
        mc_masks = []
        labels = datum['label'][:2]
        for i, label in enumerate(labels):
            binary_mask = torch.where(mask==(i+1), 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)
        mc_masks[0] += mc_masks[1]

        mask = torch.concat(mc_masks, dim=0) # [C, H, W, D]
        
        ##img = repeat(img, 'c h w d -> (c r) h w d', r=3)
        mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        # img = Normalization(img, 'MRI' )

        return img, mask, labels, datum['modality'], datum['image'], datum['mask']
    
    def PanSeg(self, datum:dict) -> tuple:
        """
        labels = [
            'pancreas',
            ]
        """
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.EnsureChannelFirstd(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                # monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                # monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image'] # [1, H, W, D]
        mask = dictionary['label']# [1, H, W, D]
        
        # original
        labels = datum['label'][:1]
        
        ##img = repeat(img, 'c h w d -> (c r) h w d', r=3)
        mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        # img = Normalization(img, 'MRI')

        return img, mask, labels, datum['modality'], datum['image'], datum['mask']
    
    def LiQA(self, datum:dict) -> tuple:
        """
        labels = [
            'liver',
            ]
        """
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.EnsureChannelFirstd(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                # monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                # monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image'] # [1, H, W, D]
        mask = dictionary['label']# [1, H, W, D]
        
        # original
        labels = datum['label'][:1]
        
        ##img = repeat(img, 'c h w d -> (c r) h w d', r=3)
        mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        # img = Normalization(img, 'MRI')

        return img, mask, labels, datum['modality'], datum['image'], datum['mask']

    def MSD_Prostate(self, datum:dict) -> tuple:
        mod2channel = {"T2":0, "ADC":1}
        tmp = datum['image'].split('/')
        mod = tmp[-1]
        channel = mod2channel[mod]
        img_path = '/'.join(tmp[:-1])
        
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label'], ensure_channel_first=True),
                #monai.transforms.EnsureChannelFirstd(keys=['image']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                # monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                # monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':img_path, 'label':datum['mask']})
        img = dictionary['image'][channel, :, :, :] # [H, W, D]
        mask = dictionary['label'] # [1, H, W, D]
        
        labels = datum['label'][:2]
        
        mc_masks = []
        for i, label in enumerate(labels):
            binary_mask = torch.where(mask==(i+1), 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)
            
        # mc_masks.append(mc_masks[0]+mc_masks[1]) 
        # labels.append('prostate')
        
        mask = torch.cat(mc_masks, dim=0) # [3, H, W, D]
        
        mask = (mask-mask.min())/(mask.max()-mask.min()) # normal到0~1之间
        mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        
        img = repeat(img, 'h w d -> c h w d', c=1)  # [C, H, W, D]
        # img = Normalization(img, 'MRI')
        
        return img, mask, labels, datum['modality'], datum['image'], datum['mask']
    
    def MSD_Liver(self, datum:dict) -> tuple:
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.EnsureChannelFirstd(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                # monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                # monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image'] # [1, H, W, D]
        mask = dictionary['label']# [C, H, W, D]
        
        labels = datum['label'][:2]
        
        mc_masks = []
        for i, label in enumerate(labels):
            binary_mask = torch.where(mask==(i+1), 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)    # [1, H, W, D]
        mask = torch.concat(mc_masks, dim=0) # [C, H, W, D]
        # 0 is liver, 1 is liver tumor, should be included in liver
        # mask[0] += mask[1]
        
        mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        # img = Normalization(img, 'CT')
        
        return img, mask, labels, datum['modality'], datum['image'], datum['mask']


class Name_Mapper():
    # /mnt/hwfile/medai/zhaoziheng/SAM/SAM/CHAOS/Train_Sets/MR/34/T1DUAL/DICOM_anon/InPhase/701_.nii.gz --> 34_T1DUAL_DICOM_anon_InPhase_701_
    
    def __init__(self):
        pass

    def CHAOS_MRI(self, img_path:str) -> str:
        target_img_filename = img_path.split('/')[-5]+'_'+img_path.split('/')[-4]+'_'+img_path.split('/')[-3]+'_'+img_path.split('/')[-2]+'_'+img_path.split('/')[-1]
        target_img_filename = target_img_filename.replace('.nii.gz', '')
        return target_img_filename
    
    def MSD_Prostate(self, img_path:str) -> str:
        target_img_filename = img_path.split('/')[-2]+'_'+img_path.split('/')[-1]   # prostate_35.nii.gz_T2
        target_img_filename = target_img_filename.replace('.nii.gz', '')    # prostate_35_T2
        return target_img_filename
    
    def AMOS22_CT(self, img_path:str) -> str:
        target_img_filename = img_path.split('/')[-1]   # amos_0033.nii.gz
        target_img_filename = target_img_filename.replace('.nii.gz', '')    # amos_0033
        return target_img_filename
    
    def MSD_Liver(self, img_path:str) -> str:
        target_img_filename = img_path.split('/')[-1]   # liver_37.nii.gz
        target_img_filename = target_img_filename.replace('.nii.gz', '')    # liver_37
        return target_img_filename
    
    def PROMISE12(self, img_path:str) -> str:
        target_img_filename = img_path.split('/')[-1]   # Case15.nii.gz
        target_img_filename = target_img_filename.replace('.nii.gz', '')    # Case15
        return target_img_filename
    
    def LiQA(self, img_path:str) -> str:
        target_img_filename = img_path.split('/')[-3]+'_'+img_path.split('/')[-2]+'_'+img_path.split('/')[-1]   # Vendor_A_1920-A-S1_GED4.nii.gz
        target_img_filename = target_img_filename.replace('.nii.gz', '')    # Vendor_A_1920-A-S1_GED4
        return target_img_filename
    
    def ATLAS(self, img_path:str) -> str:
        target_img_filename = img_path.split('/')[-1]   # im52.nii.gz
        target_img_filename = target_img_filename.replace('.nii.gz', '')    # im52
        return target_img_filename


def resize_image_itk(itkimage, newSize, resamplemethod=sitk.sitkNearestNeighbor):
    resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize()
    originSpacing = itkimage.GetSpacing()
    newSize = np.array(newSize, float)
    factor = originSize / newSize
    newSpacing = originSpacing * factor
    newSize = newSize.astype(np.int)
    resampler.SetReferenceImage(itkimage)
    resampler.SetSize(newSize.tolist())
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)
    return itkimgResampled

def save_img(slice, label, dir):
    np.savez_compressed(dir, image=slice, label=label)

def norm(slices, modality):
    
    if modality == 'CT':
        lower_bound, upper_bound = -500, 1000
        slices = np.clip(slices, lower_bound, upper_bound)
    else:
        lower_bound, upper_bound = np.percentile(slices, 1), np.percentile(slices, 99)
        slices = np.clip(slices, lower_bound, upper_bound)
    
    max = np.max(slices)
    min = np.min(slices)
    slices = 2 * (slices - min) / (max - min) - 1
    return slices

def nonlinear_transformation(slices):

    points_1 = [[-1, -1], [-1, -1], [1, 1], [1, 1]]
    xvals_1, yvals_1 = bezier_curve(points_1, nTimes=100000)
    xvals_1 = np.sort(xvals_1)

    nonlinear_slices_1 = np.interp(slices, xvals_1, yvals_1)
    nonlinear_slices_1[nonlinear_slices_1 == 1] = -1

    return slices, nonlinear_slices_1

def resize_with_padding(slice, target_size=512, nearest_mode=True):
    # 获取原始数组的高度和宽度
    h, w = slice.shape
    
    # 计算需要填充的尺寸
    if h > w:
        pad_width = (h - w) // 2
        padding = ((0, 0), (pad_width, h - w - pad_width))
        # 使用 np.pad 进行填充
        padded = np.pad(slice, padding, mode='constant', constant_values=0)
    elif w > h:
        pad_height = (w - h) // 2
        padding = ((pad_height, w - h - pad_height), (0, 0))
        # 使用 np.pad 进行填充
        padded = np.pad(slice, padding, mode='constant', constant_values=0)
    else:
        padded = slice
    
    # 确保填充后的尺寸是方形
    assert padded.shape[0] == padded.shape[1], "Padding did not result in a square matrix."
    
    # 使用 OpenCV 进行缩放（插值方式可根据需求更改）
    if nearest_mode:
        resized = cv2.resize(padded, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    else:
        resized = cv2.resize(padded, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    
    return resized    


def save_test_npz(json_file, target_root):
    
    loader = Loader_Wrapper()
    name_mapper = Name_Mapper()
    
    with open(json_file, 'r') as f:
        lines = f.readlines()
        data = [json.loads(line) for line in lines]
        
    if 'AMOS22CT_CT' in json_file or 'MSD_Liver_CT' in json_file:
        modality = 'CT'
    else:
        modality = 'MRI'

    for datum in tqdm(data):
        
        func_name = datum['dataset']
        batch = getattr(loader, func_name)(datum)
        slices, masks, text_ls, _, image_path, _ = batch
        slices = slices.numpy()
        masks = masks.numpy()
        
        # counter-clock 90 degree and flip horizontally
        slices = np.rot90(slices, axes=(1, 2))
        slices = np.flip(slices, axis=2)
        masks = np.rot90(masks, axes=(1, 2))
        masks = np.flip(masks, axis=2)
        
        # nib_img = nib.load(datum['image'])
        # nib_mask = nib.load(datum['mask'])

        # affine = nib_img.affine.copy()
        
        # slices = nib_img.get_fdata()
        # masks = nib_mask.get_fdata()
        # masks[masks != 0] = 1
        
        # convert mc mask into sc mask
        sc_mask = []
        for i, label in enumerate(text_ls):    # 0, kidney ...
            tmp = masks[i, :, :, :] * (i+1)   # --> 1
            sc_mask.append(tmp)
        sc_mask = np.stack(sc_mask, axis=0)   # (N, H, W, D)
        sc_mask = np.max(sc_mask, axis=0)   # H W D
        masks = sc_mask

        # normalize image
        slices = slices[0]
        slices = norm(slices, modality)
        
        # TODO: to calculate FID, wont use in training or testing
        slices, nonlinear_slices_1 = nonlinear_transformation(slices)
        if not os.path.exists(os.path.join(target_root, 'visual_all_sd')):
            os.makedirs(os.path.join(target_root, 'visual_all_sd'))
        # 

        if not os.path.exists(os.path.join(target_root, 'test')):
            os.makedirs(os.path.join(target_root, 'test'))
            
        image_name = getattr(name_mapper, datum['dataset'])(datum['image'])
        
        for i in range(slices.shape[2]):
            tmp_slice = slices[:, :, i]
            tmp_slice = resize_with_padding(tmp_slice, 512, nearest_mode=False)
            
            tmp_mask = masks[:, :, i]
            tmp_mask = resize_with_padding(tmp_mask, 512, nearest_mode=True)
            
            save_img(tmp_slice, tmp_mask, os.path.join(target_root, 'test', f'{image_name}_s{i}.npz'))
            
            # TODO: to calculate FID, wont use in training or testing
            tmp_nonlinear_slice = nonlinear_slices_1[:, :, i]
            tmp_nonlinear_slice = resize_with_padding(tmp_nonlinear_slice, 512, nearest_mode=False)
            assert np.max(tmp_nonlinear_slice) <= 1 and np.min(tmp_nonlinear_slice) >= -1
            visualize_nonlinear_slice = (tmp_nonlinear_slice + 1)/2 # [0,1]
            visualize_nonlinear_slice = visualize_nonlinear_slice * 255
            cv2.imwrite(os.path.join(target_root, 'visual_all_sd', f'{image_name}_s{i}_nonlinear.png'), visualize_nonlinear_slice)
            # 


def prepare_train(json_file, target_root):

    loader = Loader_Wrapper()
    name_mapper = Name_Mapper()
    
    with open(json_file, 'r') as f:
        lines = f.readlines()
        data = [json.loads(line) for line in lines]
        
    if 'AMOS22CT_CT' in json_file or 'MSD_Liver_CT' in json_file:
        modality = 'CT'
    else:
        modality = 'MRI'
        
    if not os.path.exists(os.path.join(target_root, 'train/ss')):
        os.makedirs(os.path.join(target_root, 'train/ss'))
    if not os.path.exists(os.path.join(target_root, 'train/sd')):
        os.makedirs(os.path.join(target_root, 'train/sd'))
    if not os.path.exists(os.path.join(target_root, 'visual_all_sd')):
        os.makedirs(os.path.join(target_root, 'visual_all_sd'))    
    if not os.path.exists(os.path.join(target_root, 'visual_all_ss')):
        os.makedirs(os.path.join(target_root, 'visual_all_ss'))    

    for datum in tqdm(data):
        
        func_name = datum['dataset']
        batch = getattr(loader, func_name)(datum)
        slices, masks, text_ls, _, image_path, _ = batch
        slices = slices.numpy()
        masks = masks.numpy()
        
        # counter-clock 90 degree and flip horizontally
        slices = np.rot90(slices, axes=(1, 2))
        slices = np.flip(slices, axis=2)
        masks = np.rot90(masks, axes=(1, 2))
        masks = np.flip(masks, axis=2)
        
        # nib_img = nib.load(datum['image'])
        # nib_mask = nib.load(datum['mask'])

        # affine = nib_img.affine.copy()
        
        # slices = nib_img.get_fdata()
        # masks = nib_mask.get_fdata()
        # masks[masks != 0] = 1
        
        # convert mc mask into sc mask
        sc_mask = []
        for i, label in enumerate(text_ls):    # 0, kidney ...
            tmp = masks[i, :, :, :] * (i+1)   # --> 1
            sc_mask.append(tmp)
        sc_mask = np.stack(sc_mask, axis=0)   # (N, H, W, D)
        sc_mask = np.max(sc_mask, axis=0)   # H W D
        masks = sc_mask

        # normalize image
        slices = slices[0]
        slices = norm(slices, modality)
        
        slices, nonlinear_slices_1 = nonlinear_transformation(slices)
            
        image_name = getattr(name_mapper, datum['dataset'])(datum['image'])

        for i in range(slices.shape[2]):
            tmp_slice = slices[:, :, i]
            tmp_slice = resize_with_padding(tmp_slice, 512, nearest_mode=False)
            assert np.max(tmp_slice) <= 1 and np.min(tmp_slice) >= -1
            
            tmp_nonlinear_slice = nonlinear_slices_1[:, :, i]
            tmp_nonlinear_slice = resize_with_padding(tmp_nonlinear_slice, 512, nearest_mode=False)
            assert np.max(tmp_nonlinear_slice) <= 1 and np.min(tmp_nonlinear_slice) >= -1

            if random.random() < 1:   # randomly visualization
                
                visualize_slice = (tmp_slice + 1)/2 # [0,1]
                visualize_slice = visualize_slice * 255
                cv2.imwrite(os.path.join(target_root, 'visual_all_ss', f'{image_name}_s{i}.png'), visualize_slice)

                # Save tmp_nonlinear_slice
                visualize_nonlinear_slice = (tmp_nonlinear_slice + 1)/2 # [0,1]
                visualize_nonlinear_slice = visualize_nonlinear_slice * 255
                cv2.imwrite(os.path.join(target_root, 'visual_all_sd', f'{image_name}_s{i}_nonlinear.png'), visualize_nonlinear_slice)
                        
            tmp_mask = masks[:, :, i]
            tmp_mask = resize_with_padding(tmp_mask, 512, nearest_mode=True)
            
            """
            Source-Similar
            """
            save_img(tmp_slice, tmp_mask, os.path.join(target_root, 'train/ss', f'{image_name}_s{i}.npz'))
            """
            Source-Dissimilar
            """
            save_img(tmp_nonlinear_slice, tmp_mask, os.path.join(target_root, 'train/sd', f'{image_name}_s{i}.npz'))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Compute Dice scores')
    parser.add_argument('--train_jsonl', type=str, default='/mnt/hwfile/medai/zhaoziheng/SAM/MRDiffusion/trainsets/CHAOS_MRI/CHAOS_MRI_T2_SPIR.jsonl')
    parser.add_argument('--test_jsonl', type=str, default='/mnt/hwfile/medai/zhaoziheng/SAM/MRDiffusion/testsets/CHAOS_MRI/CHAOS_MRI_T2_SPIR.jsonl')
    parser.add_argument('--target_root', type=str, default='/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/Dual-Normalization/data/CHAOS_T2SPIR')
    args = parser.parse_args()

    prepare_train(args.train_jsonl, args.target_root)

    save_test_npz(args.test_jsonl, args.target_root)
        
    """
    CHAOS_T2
    ├── train
        ├── ss
            ├── sample0.npz, sample1.npz, xxx
        └── sd
    └── test
        ├── test_sample0.npz, test_sample1.npz, xxx
    """
        