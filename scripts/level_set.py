from __future__ import division
import numpy as np
import cv2
from scipy import misc, ndimage
import torch
from torch.nn import functional as F
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import os 

def Heaviside(phi, epsilon=1):#-1/2
    pi = 3.141593
    H = 0.5 * (1 + 2/pi * torch.atan(phi/epsilon))
    return H

def Dirac(phi, epsilon:1):
    pi = 3.141593
    Delta_h = (1/pi)*epsilon*(epsilon**2 + phi**2)
    return Delta_h

def Calculate_c(Img, H_phi):
    numer_1 = torch.sum(H_phi*Img.unsqueeze(0), axis=[1,2]) 
    denom_1 = torch.sum(H_phi, axis=[1,2])
    C1 = numer_1/denom_1
    numer_2 = torch.sum((1-H_phi)*Img.unsqueeze(0), axis=[1,2])
    denom_2 = torch.sum(1-H_phi, axis=[1,2])
    C2 = numer_2/denom_2
    return C1,C2

def del2(x):
    """
        torch version del2
        x: (N, C, H, W)
        Pay attention to the signal!
    """
    assert x.dim() == 4
    laplacian = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
    laplacian = torch.FloatTensor(laplacian).unsqueeze(0).unsqueeze(0).cuda()

    x = F.conv2d(x, laplacian, padding=0)
    x = torch.nn.ReplicationPad2d(1)(x)
    return 0.25*x


def gradient(x, split=True):
    """
        returns (gy, gx) following the rules of np.gradient!
        torch version gradient for 2D
        x: (N, C, H, W)
    """
    assert len(x.shape) == 4
    [nrow, ncol] = x.shape[-2:]

    gy = x.clone()
    gy[..., 1:nrow - 1, :] = (x[..., 2:nrow, :] - x[..., 0:nrow - 2, :]) / 2
    gy[..., 0, :] = x[..., 1, :] - x[..., 0, :]
    gy[..., nrow - 1, :] = x[..., nrow - 1, :] - x[..., nrow - 2, :]

    gx = x.clone()
    gx[..., 1:ncol - 1] = (x[..., 2:ncol] - x[..., 0:ncol - 2]) / 2
    gx[..., 0] = x[..., 1] - x[..., 0]
    gx[..., ncol - 1] = x[..., ncol - 1] - x[..., ncol - 2]

    if not split:
        return torch.cat((gy, gx), dim=1)
    return gy, gx


def gradient_sobel(map, split=True):
    """
    returns (gy, gx) following the rules of np.gradient!
    :param map: (N, C=1, H, W)
    :return:
    """

    return gradient(map, split)

    # sobel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    # sobel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    #
    # sobel_x = torch.FloatTensor(sobel_x).unsqueeze(0).unsqueeze(0).cuda() / 8
    # sobel_y = torch.FloatTensor(sobel_y).unsqueeze(0).unsqueeze(0).cuda() / 8
    # sobel = torch.cat((sobel_y, sobel_x), dim=0)
    #
    # g = F.conv2d(map, sobel, padding=0)    # (N, C=2, H, W)
    # g = torch.nn.ReplicationPad2d(1)(g)
    #
    # if split:
    #     gy = g[:, 0, :, :].unsqueeze(1)
    #     gx = g[:, 1, :, :].unsqueeze(1)
    #     return gy, gx
    # return g


def curvature_central(u):
    # compute curvature
    [ux,uy] = gradient_sobel(u, split=True)
    normDu = torch.sqrt(ux**2 + uy**2 + 1e-10)  # the norm of the gradient plus a small possitive number to avoid division by zero in the following computation.
    Mx = ux / normDu
    My = uy / normDu
    [nxx, junk] = gradient_sobel(Mx)
    [junk, nyy] = gradient_sobel(My)
    k = nxx + nyy
    return k

def div(nx, ny):
    [_, nxx] = gradient_sobel(nx, split=True)
    [nyy, _] = gradient_sobel(ny, split=True)
    return nxx + nyy


def distReg_p2(phi):
    """
        compute the distance regularization term with the double-well potential p2 in equation (16)
    """
    # phi = phi.unsqueeze(dim=1)
    [phi_y, phi_x] = gradient_sobel(phi, split=True)
    s = torch.sqrt(torch.pow(phi_x, 2) + torch.pow(phi_y, 2) + 1e-10)
    a = ((s >= 0) & (s <= 1)).float()
    b = (s > 1).float()
    # compute first order derivative of the double-well potential p2 in equation (16)
    ps = a * torch.sin(2 * np.pi * s) / (2 * np.pi) + b * (s - 1)
    # compute d_p(s)=p'(s)/s in equation (10). As s-->0, we have d_p(s)-->1 according to equation (18)
    neq0 = lambda x: ((x < -1e-10) | (x > 1e-10)).float()
    eq0 = lambda x: ((x >= -1e-10) & (x <= 1e-10)).float()
    dps = (neq0(ps) * ps + eq0(ps)) / (neq0(s) * s + eq0(s))
    return div(dps * phi_x - phi_x, dps * phi_y - phi_y) + del2(phi)


def NeumannBoundCond(f):
    """
        Make a function satisfy Neumann boundary condition
    """
    N, K, H, W = f.shape

    g = f  # f.clone()
    g = torch.reshape(g, (N * K, H, W))
    [_, nrow, ncol] = g.shape

    g[..., [0, nrow - 1], [0, ncol - 1]] = g[..., [2, nrow - 3], [2, ncol - 3]]
    g[..., [0, nrow - 1], 1: ncol - 1] = g[..., [2, nrow - 3], 1: ncol - 1]
    g[..., 1: nrow - 1, [0, ncol - 1]] = g[..., 1: nrow - 1, [2, ncol - 3]]

    g = torch.reshape(g, (N, K, H, W))
    return g

def Calculate_multi_c(Img, H_phi1, H_phi2):
    M11 = H_phi1 * H_phi2  # 区域1
    M10 = H_phi1 * (1 - H_phi2)  # 区域2
    M01 = (1 - H_phi1) * H_phi2  # 区域3
    M00 = (1 - H_phi1) * (1 - H_phi2)  # 区域4

    # 计算每个区域的均值
    C11 = torch.sum(M11 * Img) / torch.sum(M11) if torch.sum(M11) != 0 else 0
    C10 = torch.sum(M10 * Img) / torch.sum(M10) if torch.sum(M10) != 0 else 0
    C01 = torch.sum(M01 * Img) / torch.sum(M01) if torch.sum(M01) != 0 else 0
    C00 = torch.sum(M00 * Img) / torch.sum(M00) if torch.sum(M00) != 0 else 0

    return C11, C10, C01, C00

def multiCV(phi1, phi2, I, epoch, epsilon = 1,timestep = 0.001,lambda_11 = 0.0001, lambda_10 = 0.0001, lambda_01 = 0.0001, lambda_00 = 0.0001, nu = 1):
    I = I.squeeze(0).cuda()
    I = normalize_to_0_255(I)
    for i in range(epoch):
        H_phi1 = Heaviside(phi1, epsilon)
        H_phi2 = Heaviside(phi2, epsilon)
        dirac_phi1 = Dirac(phi1, epsilon)
        dirac_phi2 = Dirac(phi2, epsilon)
        
        C11, C10, C01, C00 = Calculate_multi_c(I, H_phi1, H_phi2)
        f11 = ((I - C11)**2 - (I - C01)**2) * H_phi2
        f12 = ((I - C10)**2 - (I - C00)**2) * (1-H_phi2)
        f21 = ((I - C11)**2 - (I - C10)**2) * H_phi1
        f22 = ((I - C01)**2 - (I - C00)**2) * (1-H_phi1)
        # 更新phi1和phi2
        phi1 += timestep * (dirac_phi1 * ( - 0.001*(f11 + f12)))
        phi2 += timestep * (dirac_phi2 * ( - 0.001*(f21 + f22)))
    
    return phi1, phi2

def drlse_edge(phi_0, g, iter, lambda_val, mu, alfa, dirac_ratio, timestep):
    phi = phi_0.clone().unsqueeze(0).unsqueeze(0)
    vf = gradient_sobel(g.unsqueeze(0).unsqueeze(0), split=False)
    vy = vf[:, 0, :, :].unsqueeze(1)
    vx = vf[:, 1, :, :].unsqueeze(1)
    for _ in range(iter):
        phi = NeumannBoundCond(phi)
        [phi_y, phi_x] = gradient_sobel(phi, split=True)
        s = torch.sqrt(torch.pow(phi_x, 2) + torch.pow(phi_y, 2) + 1e-10)
        nx = phi_x / (s)
        ny = phi_y / (s)
        curvature = div(nx, ny)
        
        dist_reg_term = distReg_p2(phi)        
        diracPhi = Dirac(phi, dirac_ratio)
        area_term = diracPhi * g
        edge_term = diracPhi * (vx * nx + vy * ny) + diracPhi * g * curvature
        phi += timestep * (mu * dist_reg_term + lambda_val * edge_term + alfa * area_term)

    return -phi.squeeze(0).squeeze(0)

def laplacian(phi):
    lap_kernel = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], dtype=phi.dtype, device=phi.device).unsqueeze(0).unsqueeze(0)
    return torch.nn.functional.conv2d(phi, lap_kernel, padding=1)

def compute_d(T):
    """
    T: torch tensor [H,W], values in {-1,0,1}
    returns: d, torch tensor [H,W], each pixel's distance to nearest {-1,1} location
    """
    # Mask where T is -1 or 1
    mask = (T == -1) | (T == 1)
    # Convert to numpy boolean array where foreground is False, background is True
    fg = ~mask.cpu().numpy()
    # Compute distance transform
    dist = ndimage.distance_transform_edt(fg)
    return torch.from_numpy(dist).to(T.device).float()


def convex_LSF(phi_0, T, alpha, lambda1, lambda2, timestep, iter, lsf_dir):
    """
    phi_0: 初始 level set 函数, [H,W]
    T: 模板, same shape as phi_0, values in {-1,0,1}
    alpha: 权重因子, scalar or [H,W]
    lambda1, lambda2: 系数
    timestep: 更新步长
    iter: 迭代次数
    """

    phi = phi_0.clone().unsqueeze(0).unsqueeze(0).float()  # [1,1,H,W]

    T = T.unsqueeze(0).unsqueeze(0).float()
    
    if isinstance(alpha, torch.Tensor):
        alpha = alpha.unsqueeze(0).unsqueeze(0).float()
    
    os.makedirs(lsf_dir, exist_ok=True)
    for _ in range(iter):
        phi = NeumannBoundCond(phi)
        torch.save(phi, os.path.join(lsf_dir, f'phi_iter{_}.pt'))
        
        # gradient
        phi_y, phi_x = gradient_sobel(phi, split=True)
        s = torch.sqrt(phi_x**2 + phi_y**2 + 1e-10)
        nx = phi_x / s
        ny = phi_y / s

        # curvature: divergence of normalized gradient
        curvature = div(nx, ny)

        # Laplacian smoothing
        laplacian_phi = laplacian(phi)

        # compute each term
        template_force = - lambda1 * (alpha**2) * (phi - T)
        smoothing_force = lambda2 * laplacian_phi
        curvature_force = curvature

        # update        
        dphi = template_force + smoothing_force + curvature_force
        if dphi.mean() < 1e-5:
            phi += timestep * dphi
            break
        phi += timestep * dphi
    return phi.squeeze(0).squeeze(0)

def multiDRLSE(phi_U, phi_L, g, iter, lambda_val, mu, alfa, dirac_ratio, timestep):
    phi_L = phi_L.clone().unsqueeze(0).unsqueeze(0)
    phi_U = phi_U.clone().unsqueeze(0).unsqueeze(0)
    vf = gradient_sobel(g.unsqueeze(0).unsqueeze(0), split=False)
    vy = vf[:, 0, :, :].unsqueeze(1)
    vx = vf[:, 1, :, :].unsqueeze(1)
    def gradient_flow(phi, iter):
        for _ in range(iter):
            phi = NeumannBoundCond(phi)
            [phi_y, phi_x] = gradient_sobel(phi, split=True)
            s = torch.sqrt(torch.pow(phi_x, 2) + torch.pow(phi_y, 2) + 1e-10)
            nx = phi_x / (s)
            ny = phi_y / (s)
            curvature = div(nx, ny)
            
            dist_reg_term = distReg_p2(phi)        
            diracPhi = Dirac(phi, dirac_ratio)
            area_term = diracPhi * g
            edge_term = diracPhi * (vx * nx + vy * ny) + diracPhi * g * curvature
            phi += timestep * (mu * dist_reg_term + lambda_val * edge_term + alfa * area_term)
    phi_L_t = gradient_flow(phi_L, iter)
    phi_U_t = gradient_flow(phi_U, iter)
    return -phi_L_t, -phi_U_t
def prob2sdf(prob_map):
    return 2*prob_map-1

def modified_sigmoid(x, amp=1.5, z = 0.5):
    return amp / (1 + torch.exp(10 * (x - 0.5))) + z

def levelset_evolution(prob_maps, sdfs, I, m,c,r, uncertain = None, amp = 1, z = 0.5, g=None, T=5, timestep=0.1, dirac_ratio=0.3, dt_max=15, epsilon=1, _test=False, lambda_1 = 1., lambda_2= 1,nu = 1, _normalize=False):
      
    if uncertain is not None:
        un_mi = uncertain['mutual_info']
        un_en = uncertain['entropy']
        un_level = modified_sigmoid(un_mi, amp, z)*(1-un_en)
    else:
        un_level = 1

    phi_0 = prob2sdf(un_level*sdfs).unsqueeze(0).unsqueeze(0)
    # vf = gradient_sobel(sdfs.unsqueeze(0).unsqueeze(0), split=False)  

    # if _normalize:
    #     vf = F.normalize(vf, p=2, dim=1)
    # vy = vf[:, 0, :, :].unsqueeze(1)
    # vx = vf[:, 1, :, :].unsqueeze(1)

    phi = phi_0.clone()
    for k in range(T):
        phi = NeumannBoundCond(phi)
        [phi_y, phi_x] = gradient_sobel(phi, split=True)
        s = torch.sqrt(torch.pow(phi_x, 2) + torch.pow(phi_y, 2) + 1e-10)
        Nx = phi_x / s
        Ny = phi_y / s
        curvature = div(Nx, Ny) #not curvature
        diracPhi = Dirac(phi, dirac_ratio)
        # motion_term = vx * phi_x + vy * phi_y
        motion_term = torch.log(prob_maps/(1-prob_maps+1e-10))

        # H_phi = Heaviside(phi, epsilon)
        # C1, C2 = Calculate_c(I, H_phi.squeeze(dim = 1))
        # intensity_term = nu-lambda_1*(I-C1)**2+lambda_2*(I-C2)**2

        if g is None:
            phi = phi + timestep * diracPhi * (m*motion_term  + c*curvature)
            phi = phi + r * distReg_p2(phi)
        else:
            phi = phi + timestep * diracPhi * (m*motion_term  + c*curvature)
            phi = phi + 0.2 * distReg_p2(phi)
    return phi, motion_term

def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0
    
def gaussian_kernel(kernel_size, sigma):
    # 创建一个空白的核
    kernel = torch.zeros(kernel_size, kernel_size)
    center = kernel_size // 2

    # 计算高斯核
    for i in range(kernel_size):
        for j in range(kernel_size):
            x = i - center
            y = j - center
            kernel[i, j] = torch.exp(torch.tensor(-(x**2 + y**2) / (2 * sigma**2)))
    # 标准化核
    kernel = kernel / torch.sum(kernel)
    return kernel

def localBinaryFit(Img, u, KI, KONE, Ksigma, epsilon):
    # compute f1 and f2
    Hu = 0.5 * (1 + (2 / torch.pi) * torch.atan(u / epsilon))
    I = Img * Hu
    c1 = F.conv2d(Hu, Ksigma, padding = 'same')
    c2 = F.conv2d(I, Ksigma, padding = 'same')
    f1 = c2 / c1
    f2 = (KI - c2) / (KONE - c1)
    return f1, f2

def RESLS(u0, Img, w, timestep, mu, lambda1, lambda2, epsilon, numIter, sigma=3, uncertain = None, amp = 1.5, z = 0.5,):
    Img = Img.unsqueeze(0).cuda()
    Ksigma = gaussian_kernel(kernel_size = round(2*sigma)*2+1, sigma = sigma).unsqueeze(0).unsqueeze(0).cuda()
    KI = F.conv2d(Img, Ksigma, padding = 'same')
    KONE = F.conv2d(torch.ones_like(Img), Ksigma, padding = 'same')

    u = u0.unsqueeze(0).unsqueeze(0)
    gamma = 0.04 # 1+exp() eq36

    if uncertain is not None:
        un_mi = uncertain['mutual_info']
        un_en = uncertain['entropy']
        g = modified_sigmoid(un_mi, amp, z)*(1-un_en).unsqueeze(0).unsqueeze(0).cuda()
        [vx, vy] = gradient(g)
    else:
        # un_level = 1
        [Ix, Iy] = gradient(Img)
        g = 1/(1+Ix**2+Iy**2) # edge indicator
        [vx, vy] = gradient(g)
        # un_mi.cpu().numpy()

    criterior = 1
    i = 0
    while criterior > 1e-4 and i < numIter:
        i += 1
        u = NeumannBoundCond(u)
        K = curvature_central(u)
        [ux, uy] = gradient(u)
        e = torch.sqrt(ux**2 + uy**2)
        smallNumber = 1e-10
        Nx = ux/(e+smallNumber)
        Ny = uy/(e+smallNumber)
        DrcU = (epsilon/torch.pi)/(epsilon**2.+u**2)
        [f1, f2] = localBinaryFit(Img, u, KI, KONE, Ksigma, epsilon)

        s1 = (torch.sqrt(lambda1)*f1 + torch.sqrt(lambda2)*f2)/(torch.sqrt(lambda1)+torch.sqrt(lambda2))  # eq39 alpha*u1+beta*u2    
        s2 = 2/(1+torch.exp(-gamma*(s1-Img)))-1
        s3 = sign(torch.sum(f1-f2))*s2 

        A = mu*distReg_p2(u)        # regularization 
        P = 1*DrcU*(vx*Nx+vy*Ny)    # edge 
        L = 1*DrcU*g*K + w*DrcU*K   # length
        D = 2*s3*DrcU               # region
        tmp = u+timestep*(L+P+A+D)
        criterior = torch.abs(torch.sum(-tmp)-torch.sum(-u))/torch.abs(torch.sum(u))
        u = u+timestep*(L+P+A+D)
        
    return u

def multiRESLS(u1, u2, Img, w, timestep, mu, lambda1, lambda2, epsilon, numIter, sigma=3, uncertain = None, amp = 1.5, z = 0.5,):
    Img = Img.unsqueeze(0).cuda()
    Ksigma = gaussian_kernel(kernel_size = round(2*sigma)*2+1, sigma = sigma).unsqueeze(0).unsqueeze(0).cuda()
    KI = F.conv2d(Img, Ksigma, padding = 'same')
    KONE = F.conv2d(torch.ones_like(Img), Ksigma, padding = 'same')

    u1 = u1.unsqueeze(0).unsqueeze(0)
    u2 = u2.unsqueeze(0).unsqueeze(0)
    
    gamma = 0.04 # 1+exp() eq36

    if uncertain is not None:
        un_mi = uncertain['mutual_info']
        un_en = uncertain['entropy']
        g = modified_sigmoid(un_mi, amp, z)*(1-un_en).unsqueeze(0).unsqueeze(0).cuda()
        [vx, vy] = gradient(g)
    else:
        # un_level = 1
        [Ix, Iy] = gradient(Img)
        g = 1/(1+Ix**2+Iy**2) # edge indicator
        [vx, vy] = gradient(g)
        # un_mi.cpu().numpy()

    # [f1, f2] = localBinaryFit(Img, u, KI, KONE, Ksigma, epsilon)

    # s1 = (torch.sqrt(lambda1)*f1 + torch.sqrt(lambda2)*f2)/(torch.sqrt(lambda1)+torch.sqrt(lambda2))  # eq39 alpha*u1+beta*u2    
    # s2 = 2/(1+torch.exp(-gamma*(s1-Img)))-1
    # s3 = sign(torch.sum(f2-f1))*s2 # f2 or f1?

    def gradientFlow(u):
        criterior = 1
        i = 0
        while criterior > 1e-4 and i < numIter:
            i += 1
            
            K = curvature_central(u)
            [ux, uy] = gradient(u)
            e = torch.sqrt(ux**2 + uy**2)
            smallNumber = 1e-10
            Nx = ux/(e+smallNumber)
            Ny = uy/(e+smallNumber)
            DrcU = (epsilon/torch.pi)/(epsilon**2.+u**2)



            A = mu*distReg_p2(u)        # regularization 
            P = 1*DrcU*(vx*Nx+vy*Ny)    # edge 
            L = 1*DrcU*g*K + w*DrcU*K   # length
            # D = 2*s3*DrcU               # region
            tmp = u+timestep*(L+P+A)

            criterior = torch.abs(torch.sum(-tmp)-torch.sum(-u))/torch.abs(torch.sum(u))
            u = u+timestep*(L+P+A)
        return u
    u1 = NeumannBoundCond(u1)
    u2 = NeumannBoundCond(u2)
    u1 = gradientFlow(u1)
    u2 = gradientFlow(u2)

    return u1, u2

def levelset_evolution_probasphi0(prob_maps, sdfs, I, m,c,r, uncertain = None, amp = 1, z = 0.5, g=None, T=15, timestep=0.1, dirac_ratio=0.3, dt_max=15, epsilon=1, _test=False, lambda_1 = 1., lambda_2= 1,nu = 1, _normalize=False):
      
    if uncertain is not None:
        un_mi = uncertain['mutual_info']
        un_en = uncertain['entropy']
        un_level = modified_sigmoid(un_mi, amp, z)*(1-un_en)
    else:
        un_level = 1

    phi_0 = prob2sdf(un_level*prob_maps).unsqueeze(0).unsqueeze(0)
    vf = gradient_sobel(un_level*sdfs.unsqueeze(0).unsqueeze(0), split=False)  

    if _normalize:
        vf = F.normalize(vf, p=2, dim=1)
    vy = vf[:, 0, :, :].unsqueeze(1)
    vx = vf[:, 1, :, :].unsqueeze(1)

    phi = phi_0.clone()
    for k in range(T):
        phi = NeumannBoundCond(phi)
        [phi_y, phi_x] = gradient_sobel(phi, split=True)
        s = torch.sqrt(torch.pow(phi_x, 2) + torch.pow(phi_y, 2) + 1e-10)
        Nx = phi_x / s
        Ny = phi_y / s
        curvature = div(Nx, Ny)
        diracPhi = Dirac(phi, dirac_ratio)
        motion_term = vx * phi_x + vy * phi_y

        # H_phi = Heaviside(phi, epsilon)
        # C1, C2 = Calculate_c(I, H_phi.squeeze(dim = 1))
        # intensity_term = nu-lambda_1*(I-C1)**2+lambda_2*(I-C2)**2

        if g is None:
            phi = phi + timestep * diracPhi * (m*un_level*motion_term  + c*curvature)
            phi = phi + r * distReg_p2(phi)
        else:
            phi = phi + timestep * diracPhi * (m*motion_term  + c*curvature)
            phi = phi + 0.2 * distReg_p2(phi)
    return phi, un_level

def normalize_to_0_255(tensor):
    # 找到张量中的最小值和最大值
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    
    # 将最小值移到 0
    normalized_tensor = tensor - min_val
    
    # 缩放以保持相对比例
    normalized_tensor = normalized_tensor * (255 / (max_val - min_val))
    
    # 四舍五入到整数类型
    normalized_tensor = normalized_tensor.round().to(torch.uint8)
    
    return normalized_tensor

def main():

    import os
    import nibabel

    id_patient = '345'
    id_slice = '36'

    file_name_ref = id_patient+"_"+id_slice+"/"+"brats_test_"+id_patient+"_seg_"+id_slice+"_w.nii.gz"
    file_name_orig = id_patient+"_"+id_slice+"/"+"brats_test_"+id_patient+"_t1_"+id_slice+"_w.nii.gz"
    file_name_result = id_patient+"_"+id_slice+'_output'

    path_to_pred = ""
    path_to_gt = ""

    path_to_result = os.path.join(path_to_pred, file_name_result)
    path_to_reference = os.path.join(path_to_gt, file_name_ref)
    path_to_orig = os.path.join(path_to_gt, file_name_orig)
    orig = torch.tensor(nibabel.load(path_to_orig).get_fdata())[8:-8, 8:-8].cuda().float()
    img = normalize_to_0_255(orig)
    result_ = torch.load(path_to_result).cuda().float()
    prob_maps = torch.clamp(result_[:,0,:,:], min=0)
    sdfs = result_[:,1,:,:] #[-1,1] 
    prob_maps_mean = torch.mean(prob_maps, axis=0)
    sdfs_mean = torch.mean(sdfs, axis=0)
    uncertain_prob = get_uncertainty(prob_maps)
    params = (1,4,0.2)
    phi_t, motion_term = levelset_evolution_probasphi0(prob_maps_mean, sdfs_mean, img, T=10, amp = 3, uncertain=uncertain_prob, *params)
    x = np.arange(0, 224)
    y = np.arange(0, 224)
    X, Y = np.meshgrid(x, y)
    plt.imshow(tonp(phi_t),vmin=-1,vmax=1)
    plt.colorbar()
    plt.contour(X, Y, tonp(phi_t), levels=[0], colors='red', linewidths=1)
if __name__ == "__main__":
    main()

