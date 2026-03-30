from operator import index
from get_scores import dc, jc, hd95
import os
import numpy as np
import torch
import nibabel
import torch
from utils import get_uncertainty, create_diff_mask, create_pred_mask_w_narrow_band, save_concat_images
import matplotlib.pyplot as plt
from level_set import gradient_sobel
from level_set import levelset_evolution, levelset_evolution_probasphi0, RESLS, multiCV, drlse_edge, convex_LSF, compute_d
import pickle
import cv2
from torchvision import transforms
from PIL import Image
import torchvision
import scipy.ndimage as ndi
from sklearn.metrics import f1_score
import csv
from utils_my import get_gt_mask, get_res, normality_test
import scipy.stats as stats
from tqdm import tqdm
from matplotlib.patches import Patch

def normalize(self, data, mean, std):
    foreground_mask = data != 0
    normalized_data = data.clone()
    normalized_data[foreground_mask] = (normalized_data[foreground_mask] - mean) / std
    return normalized_data

def prob2sdf(prob_map, scale = 2):
    return scale*(2*prob_map-1)#+0.5*scale

def compute_confidence_intervals(sdfs, confidence=0.95):
    if torch.isnan(sdfs.any()):
        raise ValueError("sdfs contains NaN values")
    # 计算均值和标准误
    mean = sdfs.mean(dim=0)  # [H, W]
    sem = sdfs.std(dim=0, correction=1) / (sdfs.size(0) ** 0.5)  # 标准误 [H, W]
    # 计算 t 临界值
    t_critical = stats.t.ppf((1 + confidence) / 2, sdfs.size(0) - 1)
    # 计算置信区间
    margin_of_error = sem * t_critical
    ci_lower = mean - margin_of_error  # 下限 [H, W]
    ci_upper = mean + margin_of_error  # 上限 [H, W]

    return mean, ci_lower, ci_upper

from scipy.stats import gaussian_kde

import torch
import numpy as np
from scipy.stats import gaussian_kde

import torch
import numpy as np
from scipy.stats import gaussian_kde
from itertools import product

def hpb_mfv_estimation_tensor(data_tensor: torch.Tensor, mask_matrix, B: int = 1000, ci_percentiles=(2.5, 97.5)):
    """
    对任意形状的输入数据进行 HPB 和 MFV 估计。

    参数：
        data_tensor: torch.Tensor
            输入数据，形状为 (T, *spatial_dims)，其中 T 是采样次数。
        B: int
            自助采样次数，默认值为 1000。
        ci_percentiles: tuple
            置信区间的百分位数，默认值为 (2.5, 97.5)。

    返回：
        mfv_map: torch.Tensor
            每个变量的最频值估计，形状为 (*spatial_dims)。
        ci_lower: torch.Tensor
            每个变量置信区间的下限，形状为 (*spatial_dims)。
        ci_upper: torch.Tensor
            每个变量置信区间的上限，形状为 (*spatial_dims)。
    """
    # 验证输入数据的维度
    assert data_tensor.ndim >= 2, "输入数据的维度应大于等于 2。"

    # 获取采样次数和空间维度
    T = data_tensor.shape[0]
    spatial_dims = data_tensor.shape[1:]

    # 计算每个变量的均值和标准差
    mu = data_tensor.mean(dim=0)  # 形状为 (*spatial_dims)
    sigma = data_tensor.std(dim=0, correction=1)  # 形状为 (*spatial_dims)

    # 扩展维度以进行广播
    mu_expanded = mu.unsqueeze(0)  # 形状为 (1, *spatial_dims)
    sigma_expanded = sigma.unsqueeze(0)  # 形状为 (1, *spatial_dims)

    # 进行混合参数自助采样
    samples = torch.normal(mu_expanded.expand(B, *spatial_dims), sigma_expanded.expand(B, *spatial_dims))  # 形状为 (B, *spatial_dims)

    # 初始化 MFV 和置信区间数组
    mfv_map, ci_lower, ci_upper = compute_confidence_intervals(data_tensor, confidence = 0.95)

    # 获取所有空间位置的索引
    spatial_indices = list(product(*[range(s) for s in spatial_dims]))

    # 对每个变量进行估计
    for idx in tqdm(spatial_indices, desc="Estimating MFV and CI"):
        if mask_matrix[idx] == 0:
            continue
        pixel_samples = samples[(slice(None),) + idx].cpu().numpy()

        # 使用核密度估计估计概率密度函数
        kde = gaussian_kde(pixel_samples)
        x_grid = np.linspace(pixel_samples.min(), pixel_samples.max(), 1000)
        kde_values = kde.evaluate(x_grid)
        mfv_map[idx] = x_grid[np.argmax(kde_values)]

        # 计算置信区间
        ci_lower[idx] = np.percentile(pixel_samples, ci_percentiles[0])
        ci_upper[idx] = np.percentile(pixel_samples, ci_percentiles[1])

    return mfv_map, ci_lower, ci_upper

import numpy as np
from scipy.ndimage import uniform_filter

def smooth_images(data, kernel_size=3):
    # data: (n_samples, H, W)
    smoothed = np.array([uniform_filter(img, size=kernel_size) for img in data])
    return smoothed

def bootstrap_ci(data, mask_matrix, B=1000, alpha=0.05):
    """
    data: shape (n_samples, H, W) - after smoothing
    Returns: ci_lower, ci_upper of shape (H, W)
    """
    n_samples, h, w = data.shape
    ci_lower = np.zeros((h, w))
    ci_upper = np.zeros((h, w))

    for i in range(h):
        for j in range(w):
            if mask_matrix[i, j] == 0:
                continue
            pixel_values = data[:, i, j]
            bootstrap_means = np.empty(B)
            for b in range(B):
                sample = np.random.choice(pixel_values, size=n_samples, replace=True)
                bootstrap_means[b] = np.mean(sample)
            ci_lower[i, j] = np.percentile(bootstrap_means, 100 * alpha / 2)
            ci_upper[i, j] = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    return torch.tensor(ci_lower).cuda(), torch.tensor(ci_upper).cuda()

import torch
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

def save_tensor_as_jpeg(tensor, filename, path_to_pred):
    os.makedirs(path_to_pred, exist_ok=True)

    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()

    if tensor.ndim == 2:
        unique_vals = np.unique(tensor)
        if np.all(np.isin(unique_vals, [-1,0,1])):
            cmap = ListedColormap(['blue', 'yellow', 'red'])  # +1, 0, -1
            bounds = [-1.5, -0.5, 0.5, 1.5]
            norm = BoundaryNorm(bounds, cmap.N)

            fig, ax = plt.subplots(figsize=(6, 6))
            im = ax.imshow(tensor, cmap=cmap, norm=norm)
            ax.axis('off')

            legend_elements = [
                Patch(facecolor='blue', edgecolor='black', label='Certain Background ($T = +1$)'),
                Patch(facecolor='yellow', edgecolor='black', label='Ambiguous Boundary ($T = 0$)'),
                Patch(facecolor='red', edgecolor='black', label='Certain Foreground ($T = -1$)')
            ]

            legend = ax.legend(
                handles=legend_elements,
                loc='lower center',
                bbox_to_anchor=(0.5, -0.025),  # 精调：-0.03/-0.025/-0.02 都可以试试
                fontsize=18,
                ncol=1
            )

            save_path = os.path.join(path_to_pred, filename)
            fig.savefig(
                save_path,
                dpi=300,
                format='jpeg',
                bbox_inches='tight',
                pad_inches=0.0  # 已对齐，无需多余 padding
            )

            plt.close()
            print(f"Saved three-region label map as JPEG: {save_path}")
        else: # to check TBD
            if np.min(tensor) < -1:
                cmap = ListedColormap(['blue', 'red'])
                bounds = [-5, 0, 5]
                norm = BoundaryNorm(bounds, cmap.N)
                # 创建图
                fig, ax = plt.subplots(figsize=(6, 6))
                im = ax.imshow(tensor, cmap=cmap, norm=norm,vmin=-3, vmax=3)
                ax.axis('off')
                ax.axis('off')
                legend_elements = [
                    Patch(facecolor='blue', edgecolor='black', label='Background ($\phi > 0$)'),
                    Patch(facecolor='red', edgecolor='black', label='Foreground ($\phi < 0$)')
                ]

                # 添加图例并稍微下移
                legend = ax.legend(
                    handles=legend_elements,
                    loc='lower center',
                    bbox_to_anchor=(0.5, -0.025),  
                    fontsize=18,
                    ncol=1
                )
                save_path = os.path.join(path_to_pred, filename)
                # plt.savefig(save_path, dpi=300, format='jpeg', bbox_inches='tight', pad_inches=0)
                fig.savefig(
                save_path,
                dpi=300,
                format='jpeg',
                bbox_inches='tight',
                pad_inches=0.0 
            )
                plt.close()

                print(f"Saved tensor value range [-3,+3] as JPEG: {save_path}")
            else:
                # 普通灰度图
                tensor = (tensor * 255).astype(np.uint8)
                img = Image.fromarray(tensor, mode='L')
                img.save(os.path.join(path_to_pred, filename), format='JPEG')
                print(f"Saved grayscale image as JPEG: {os.path.join(path_to_pred, filename)}")

    elif tensor.ndim == 3 and tensor.shape[0] == 3:
        # RGB 图，C,H,W -> H,W,C
        tensor = np.transpose(tensor, (1, 2, 0))
        tensor = (tensor * 255).astype(np.uint8)
        img = Image.fromarray(tensor, mode='RGB')
        img.save(os.path.join(path_to_pred, filename), format='JPEG')
        print(f"Saved RGB image as JPEG: {os.path.join(path_to_pred, filename)}")
    else:
        raise ValueError(f"Unsupported tensor shape for saving as JPEG: {tensor.shape}")

def eval(path_to_pred, path_to_gt, target_type, index, T = 15, dataset = 'BRATS', LSF = True, threshold = 0.5, print_score = False, max_cases=None):
    dices = []
    jcs = []
    hd95s = []

    case_count = 0
    stop = False
    folder_path = ""
    os.makedirs(folder_path, exist_ok=True)
    [os.unlink(os.path.join(folder_path, f)) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    for root, dirs, files in os.walk(path_to_pred):
        if not dirs:
            # files.sort()
            for f in files:
                
                gt_mask, id_patient, id_slice = get_gt_mask(dataset, root, f, path_to_gt)
                res, res_mean  = get_res(os.path.join(root, f), target_type) #res could be sdf or label
                
                norm_matrix = normality_test(res)
                
                # smoothed_norm_matrix = np.where(smooth_images(norm_matrix, kernel_size=3) >= 0, 0, 1)
                norm_matrix_mask = torch.where(torch.tensor(norm_matrix).cuda() >= 0, 0, 1)
                sdfs = res if torch.min(res) < -0.5 else prob2sdf(res, scale = 1) 

                # Generate filename
                if dataset == 'BRATS':
                    filename = f"{id_patient}_{id_slice}_na.png" #
                elif dataset == 'ISIC':
                    filename = f"ISIC_{id_patient}_na.png" #
                elif dataset == 'REFUGE2Cup':
                    filename = f"{id_patient}.png"
                else:
                    raise ValueError(f"Unknown dataset: {dataset}")
                
                
                if LSF:                   
                    phi_0 = 2*torch.mean(sdfs, dim=0)
                    mfv_map, ci_lower, ci_upper = hpb_mfv_estimation_tensor(sdfs, norm_matrix_mask)
                    
                    T = torch.zeros_like(ci_upper)
                    T[ci_upper < 0] = -1
                    T[ci_lower > 0] = 1
                    d = compute_d(T)
                    
                    uncertainty_score = get_uncertainty(res)
                    un_mi = uncertainty_score['mutual_info']
                    un_en = uncertainty_score['entropy']
                    u = (un_en)/(1+torch.exp(un_mi+0.5)) # un_en
                    g = 1-u
                    alpha = 1-d*g

                    mask_upper = torch.where(ci_upper >= 0, 1, 0)
                    mask_lower = torch.where(ci_lower >= 0, 1, 0)

                    mask_upper_lower = torch.where(mask_upper != mask_lower, 1, 0)
                
                    narrow_band = mask_upper-mask_lower
                    confidence_dir = os.path.join(os.path.dirname(path_to_pred), 'confidence_map', os.path.basename(path_to_pred),f)
                    os.makedirs(confidence_dir, exist_ok=True)

                    save_tensor_as_jpeg(ci_lower, f'ci_lower_f{f}.jpg', confidence_dir)
                    save_tensor_as_jpeg(ci_upper, f'ci_upper_f{f}.jpg', confidence_dir)
                    save_tensor_as_jpeg(mask_lower, f'mask_lower_f{f}.jpg', confidence_dir)
                    save_tensor_as_jpeg(mask_upper, f'mask_upper_f{f}.jpg', confidence_dir)
                    save_tensor_as_jpeg(narrow_band, f'narrow_band_f{f}.jpg', confidence_dir)
                    save_tensor_as_jpeg(T, f'Template_anchor{f}.jpg', confidence_dir)
                    
                    sdfs_mean = torch.mean(sdfs, dim=0)
                    binary_map = torch.where(sdfs_mean >= 0, 1, 0)
                    diff_phibar_img = create_diff_mask(binary_map, gt_mask)
                    save_tensor_as_jpeg(diff_phibar_img, f'diff_phibar_{f}.jpg', confidence_dir)
                    
                    test_parameter = True
                    print(f'******test_parameter: {test_parameter}*********\n******************')
                    
                    lsf_dir = os.path.join(os.path.dirname(path_to_pred), 'lsf', os.path.basename(path_to_pred),f)
                    if test_parameter:
                        
                        for lambda1 in np.arange(0,1.2,0.2):
                            for lambda2 in np.arange(0,1.2,0.2):
                                for iteration in np.array([ 1, 10, 20, 30]):
                                    phi_t = convex_LSF(phi_0, T, alpha, lambda1=lambda1, lambda2=lambda2, timestep=0.01, iter=iteration, lsf_dir=lsf_dir)
                    else:    
                        lambda1 = 0.2
                        lambda2 = 0.2
                        iteration = 10
                        phi_t = convex_LSF(phi_0, T, alpha, lambda1=lambda1, lambda2=lambda2, timestep=0.1, iter=iteration, lsf_dir=lsf_dir)
                    pred_mask = torch.where(phi_t >= 0, 1, 0)
                    image_pred_mask = Image.fromarray((pred_mask.cpu().detach().numpy() * 255).astype('uint8'))
                    # Ensure the target folder exists
                    output_path_pred_mask = os.path.join(f'{path_to_pred}_prediction_{lambda1}_{lambda2}_{iteration}', filename)
                    os.makedirs(os.path.dirname(output_path_pred_mask), exist_ok=True)
                    image_pred_mask.save(output_path_pred_mask)                        
                        
                    save_tensor_as_jpeg(phi_0, f'phi_0_f{f}.jpg', confidence_dir)
                    save_tensor_as_jpeg(phi_t, f'phi_t_f{f}.jpg', confidence_dir)                        
                    pred_mask = torch.where(phi_t >= 0, 1, 0)
                    pred_mask_old = torch.where(res_mean > threshold, 1, 0)
                                        
                    pictures_dir = os.path.join(os.path.dirname(path_to_pred), 'pictures', os.path.basename(path_to_pred),f)
                    save_tensor_as_jpeg(pred_mask, f'pred_mask_f{f}.jpg', pictures_dir)
                    save_tensor_as_jpeg(pred_mask_old, f'pred_mask_old_f{f}.jpg', pictures_dir)
                    save_tensor_as_jpeg(gt_mask, f'gt_mask_f{f}.jpg', pictures_dir)
                    
                    pred_mask_w_narrow_band = create_pred_mask_w_narrow_band(pred_mask, narrow_band)
                    diff_mask_img = create_diff_mask(pred_mask, gt_mask)

                    # 存储
                    save_tensor_as_jpeg(pred_mask_w_narrow_band, f'pred_mask_w_narrow_band_{f}.jpg', confidence_dir)
                    save_tensor_as_jpeg(diff_mask_img, f'diff_mask_f{f}.jpg', confidence_dir)  
                    save_concat_images(pred_mask_w_narrow_band, diff_mask_img, f'concat_pred_diff_{f}.jpg', confidence_dir)                        

                    hd95_score_old = hd95(pred_mask_old, gt_mask)

                else: # LSF = False
                    pred_mask = torch.where(res_mean > threshold, 1, 0)
                    image_pred_mask = Image.fromarray((pred_mask.cpu().detach().numpy() * 255).astype('uint8'))
                    output_path_pred_mask = os.path.join(f'{path_to_pred}_prediction', filename)
                    os.makedirs(os.path.dirname(output_path_pred_mask), exist_ok=True)
                    image_pred_mask.save(output_path_pred_mask)

                case_count += 1

                if max_cases is not None and case_count > max_cases:
                    stop = True
                    break

                dice_score = dc(pred_mask, gt_mask)
                jc_score = jc(pred_mask, gt_mask)
                hd95_score = hd95(pred_mask, gt_mask)
                if dice_score > 0.9 and hd95_score < 10:
                    print('good prediction')
                # save predictions figure
                image_pred_mask = Image.fromarray((pred_mask.cpu().detach().numpy() * 255).astype('uint8'))

                dices.append(dice_score)
                jcs.append(jc_score)
                hd95s.append(hd95_score)      
                print(f"dice: {np.mean(dices) * 100:.2f}, IoU: {np.mean(jcs)*100:.2f}, hd95s: {np.mean(hd95s):.2f}")
            if stop:
                break
    if print_score:
        print(f"total samples: {len(dices)}")
        print(f"dice: {np.mean(dices) * 100:.2f}, IoU: {np.mean(jcs)*100:.2f}, hd95s: {np.mean(hd95s):.2f}")
        
    return np.mean(dices) * 100, np.mean(jcs), np.mean(hd95s)

    
if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'

    LSF = True

    dataset = 'BRATS'
    path_to_pred = ""
    path_to_gt = ""
    

    
    target_type = path_to_pred.split('/')[-2].split('-')[0]
    print(LSF, path_to_pred, target_type)
    
    if dataset == 'REFUGE2': # multi_class label
        for i in reversed(range(2)):
            eval(path_to_pred, path_to_gt, target_type, i, T=15, dataset=dataset, LSF=LSF, threshold=0.5, print_score=True)
    else:
        eval(path_to_pred, path_to_gt, target_type, 0,  T=15, dataset=dataset, LSF=LSF, threshold=0.5, print_score=True, max_cases=10)