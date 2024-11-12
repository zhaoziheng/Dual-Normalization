import os
import argparse
import numpy as np
import medpy.metric.binary as mmb

from tqdm import tqdm
from PIL import Image
from model.unetdsbn import Unet2D
from utils.palette import color_map
from datasets.dataset import Dataset, ToTensor, CreateOnehotLabel

import torch
import torchvision.transforms as tfs
from torch.nn import DataParallel
from torch.nn import PairwiseDistance
from torch.utils.data import DataLoader
import logging
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

dataset2label_mapping = {
    'CHAOSMR' : {
            "liver": 1,
            "right kidney": 2,
            "left kidney": 3,
            "spleen": 4,
            "kidney": [2, 3]
    },
    'AMOS22_CT' : {
            "liver": 6,
            "right kidney": 2,
            "left kidney": 3,
            "spleen": 1,
            "kidney": [2, 3]
    },
    'MSD_Prostate' : {
        "transition zone of prostate": 1,
        "peripheral zone of prostate": 2,
        "prostate": [1, 2],
    },
    'Prostate' : {
        "prostate": 1,
    },
    'Liver' : {
        "liver": 1,
    },
    'Pancreas' : {
        "pancreas": 1,
    },
    'Liver Tumor' : {
        "liver": [1,2],
        "liver tumor":2,
    },
}

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./data/brats/npz_data')
parser.add_argument('--n_classes', type=int, default=2)
parser.add_argument('--test_domain_list', nargs='+', type=str)
parser.add_argument('--model_dir', type=str,  default='./results/unet_dn/model', help='model_dir')
parser.add_argument('--batch_size', type=int,  default=32)
parser.add_argument('--pred_label_dir', type=str,  default=None)
# parser.add_argument('--visual_label_dir', type=str,  default=None)
# parser.add_argument('--gt_label_dir', type=str,  default=None)
parser.add_argument('--input_image_dir', type=str,  default=None)
parser.add_argument('--gpu_ids', type=str,  default='0', help='GPU to use')
FLAGS = parser.parse_args()

def get_bn_statis(model, domain_id):
    means = []
    vars = []
    for name, param in model.state_dict().items():
        if 'bns.{}.running_mean'.format(domain_id) in name:
            means.append(param.clone())
        elif 'bns.{}.running_var'.format(domain_id) in name:
            vars.append(param.clone())
    return means, vars


def cal_distance(means_1, means_2, vars_1, vars_2):
    pdist = PairwiseDistance(p=2)
    dis = 0
    for (mean_1, mean_2, var_1, var_2) in zip(means_1, means_2, vars_1, vars_2):
        dis += (pdist(mean_1.reshape(1, mean_1.shape[0]), mean_2.reshape(1, mean_2.shape[0])) + pdist(var_1.reshape(1, var_1.shape[0]), var_2.reshape(1, var_2.shape[0])))
    return dis.item()


def visualization(pred_y, mask, fig_path):
    # Define color maps with fixed color mappings: 0 (black), 1 (blue), 2 (green), 3 (red), 4 (yellow)
    colors = ['black', 'blue', 'green', 'red', 'yellow']
    cmap = ListedColormap(colors)

    # Create a figure and a set of subplots
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns

    # Use imshow to display the mask with the specified colormap in the first subplot
    im_mask = axs[0].imshow(mask, cmap=cmap, vmin=0, vmax=4)
    axs[0].set_title('Mask')

    # Use imshow to display pred_y with the specified colormap in the second subplot
    im_pred_y = axs[1].imshow(pred_y, cmap=cmap, vmin=0, vmax=4)
    axs[1].set_title('Pred_y')

    # Create colorbars for both subplots with fixed boundaries and ticks
    cbar_mask = fig.colorbar(im_mask, ax=axs[0], ticks=[0, 1, 2, 3, 4], boundaries=[0, 1, 2, 3, 4, 5])
    cbar_pred_y = fig.colorbar(im_pred_y, ax=axs[1], ticks=[0, 1, 2, 3, 4], boundaries=[0, 1, 2, 3, 4, 5])

    # Set the labels for the colorbars
    cbar_mask.ax.set_yticklabels(['0', '1', '2', '3', '4'])
    cbar_pred_y.ax.set_yticklabels(['0', '1', '2', '3', '4'])

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(fig_path)
    

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("result.log"),
                        logging.StreamHandler()
                    ])
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_ids
    model_dir = FLAGS.model_dir
    n_classes = FLAGS.n_classes
    test_domain_list = FLAGS.test_domain_list
    num_domain = len(test_domain_list)
    print('Start Testing.')
    
    if FLAGS.pred_label_dir is not None and not os.path.exists(FLAGS.pred_label_dir):
        os.mkdir(FLAGS.pred_label_dir)
        
    if FLAGS.input_image_dir is not None and not os.path.exists(FLAGS.input_image_dir):
        os.mkdir(FLAGS.input_image_dir)
    
    # TODO: Debug in CHAOSMR ###########################################################  
    
    # if FLAGS.gt_label_dir is not None and not os.path.exists(FLAGS.gt_label_dir):
    #     os.mkdir(FLAGS.gt_label_dir)
        
    # if FLAGS.visual_label_dir is not None and not os.path.exists(FLAGS.visual_label_dir):
    #     os.mkdir(FLAGS.visual_label_dir)
        
    # TODO: Debug in CHAOSMR ########################################################### 
    
    for test_idx in range(num_domain):
        model = Unet2D(num_classes=n_classes, num_domains=2, norm='dsbn')
        model.load_state_dict(torch.load(os.path.join(model_dir, 'final_model.pth')))
        model = DataParallel(model).cuda()
        means_list = []
        vars_list = []
        for i in range(2):
            means, vars = get_bn_statis(model, i)
            means_list.append(means)
            vars_list.append(vars)

        model.train()
        dataset = Dataset(base_dir=FLAGS.data_dir, split='test', domain_list=test_domain_list[test_idx],
                        transforms=tfs.Compose([
                            CreateOnehotLabel(num_classes=FLAGS.n_classes),
                            ToTensor()
                        ]))
        dataloader = DataLoader(dataset, batch_size=FLAGS.batch_size, shuffle=False, num_workers=8, pin_memory=True)
        tbar = tqdm(dataloader, ncols=150)
        total_dice = {v:0 for v in range(1, n_classes)}
        count = 0
        with torch.no_grad():
            for idx, (batch, id) in enumerate(tbar):
                sample_data = batch['image'].cuda()
                onehot_mask = batch['onehot_label'].detach().numpy()
                mask = batch['label'].detach().numpy()
                dis = 99999999
                best_out = None
                for domain_id in range(2):
                    # model.load_state_dict(torch.load(os.path.join(model_dir, 'epoch_9.pth')))
                    output = model(sample_data, domain_label=domain_id*torch.ones(sample_data.shape[0], dtype=torch.long))
                    means, vars = get_bn_statis(model, domain_id)
                    new_dis = cal_distance(means, means_list[domain_id], vars, vars_list[domain_id])
                    if new_dis < dis or best_out == None:
                        best_out = output
                        dis = new_dis
                
                output = best_out
                pred_y = output.cpu().detach().numpy()
                pred_y = np.argmax(pred_y, axis=1)

                # if pred_y.sum() == 0 or mask.sum() == 0:
                #     total_dice += 0
                #     total_hd += 100
                #     total_asd += 100
                # else:
                #     total_dice += mmb.dc(pred_y, mask)
                #     total_hd += mmb.hd95(pred_y, mask)
                #     total_asd += mmb.asd(pred_y, mask)

                # logging.info('Domain: {}, Dice: {}, HD: {}, ASD: {}'.format(
                #     test_domain_list[test_idx],
                #     round(100 * total_dice / (idx + 1), 2),
                #     round(total_hd / (idx + 1), 2),
                #     round(total_asd / (idx + 1), 2)
                # ))
                
                # Calculate Metrics
                        
                # for i in range(pred_y.shape[0]):
                #     for v in range(1, n_classes):
                #         tmp_pred = np.where(pred_y[i]==v, 1.0, 0.0)
                #         tmp_mask = np.where(mask[i]==v, 1.0, 0.0)
                #         if pred_y.sum() == 0 and mask.sum() == 0:
                #             total_dice[v] += 1
                #         elif pred_y.sum() == 0 and mask.sum() != 0:
                #             total_dice[v] += 0
                #         elif pred_y.sum() != 0 and mask.sum() == 0:
                #             total_dice[v] += 0
                #         else:   
                #             total_dice[v] += mmb.dc(tmp_pred, tmp_mask)
                #     count += 1

                # Save Pred
                
                sample_data = sample_data[:, 0, :, :]   # b1hw -> bhw
                
                for i in range(pred_y.shape[0]):
                    image_name = id[i]
                    
                    if FLAGS.pred_label_dir is not None:
                        cv2.imwrite(os.path.join(FLAGS.pred_label_dir, image_name+'.png'), pred_y[i])
                    
                    if FLAGS.input_image_dir is not None:
                        tmp_slice = sample_data[i].cpu().detach().numpy()
                        tmp_slice = (tmp_slice - np.min(tmp_slice)) / (np.max(tmp_slice) - np.min(tmp_slice)) * 255 # in preprocess_func, image is scaled to -1~1, so we need to rescale it back to 0~255
                        cv2.imwrite(os.path.join(FLAGS.input_image_dir, image_name+'_0000.png'), tmp_slice)
            
                    # TODO: Debug in CHAOSMR ########################################################### 
                    
                    # if FLAGS.gt_label_dir is not None:
                    #    cv2.imwrite(os.path.join(FLAGS.gt_label_dir, image_name+'.png'), mask[i])
                            
                    # if FLAGS.visual_label_dir is not None:
                        # tmp_dice = int(mmb.dc(pred_y[i], mask[i]) * 100)
                        # visualization(pred_y[i], mask[i], os.path.join(FLAGS.visual_label_dir, f'({tmp_dice}){image_name}.png'))
                            
                    # TODO: Debug in CHAOSMR ########################################################### 
                        
    # for v, dice in total_dice.items():
    #     print(f'Class {v}, Dice {dice/count}')                    
                        