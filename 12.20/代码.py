import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from matplotlib import pyplot as plt

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def grad():
    grad_x_k = torch.tensor([[1,0,-1],[1,0,-1],[1,0,-1]]).repeat(3, 1, 1, 1).contiguous()
    grad_y_k = torch.tensor([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]).repeat(3, 1, 1, 1).contiguous()
    return grad_x_k, grad_y_k

def canny(img1, img2, window_size=11, canny_l1 = False, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    return _canny(img1, img2, window, window_size, channel, canny_l1, size_average)

def _canny(img1, img2, window, window_size, channel, canny_l1, size_average):
    #高斯滤波模糊
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    #求梯度
    grad_x_k, grad_y_k = grad()
    edgs1 = _canny_(mu1, channel, grad_x_k, grad_y_k, window_size)
    edgs2 = _canny_(mu2, channel, grad_x_k, grad_y_k, window_size)

    sub = torch.abs(edgs2 - edgs1)
    high_thresh = 0.6
    sub[sub > high_thresh] = 1
    return sub.mean()

#求梯度的具体步骤
def _canny_(img, channel, grad_x_k, grad_y_k, window_size):
    if img.is_cuda:
        grad_x_k = grad_x_k.cuda(img.get_device())
        grad_y_k = grad_y_k.cuda(img.get_device())
    grad_x_k = grad_x_k.type_as(img)
    grad_y_k = grad_y_k.type_as(img)
    grad_x = F.conv2d(img, grad_x_k, padding=1, groups=channel)
    grad_y = F.conv2d(img, grad_y_k, padding=1, groups=channel)

    #梯度归一化
    or_grad = torch.sqrt(grad_x ** 2 + grad_y ** 2)
    max_values, _ = torch.max(or_grad.view(3, -1), dim=1)
    max_values = max_values.unsqueeze(-1).unsqueeze(-1)
    or_grad = or_grad / max_values

    # 阈值抑制
    low_thresh = 0.1
    or_grad[or_grad < low_thresh] = 0
    #plt.imshow(or_grad.cpu().permute(1, 2, 0).numpy())  # 转换回 numpy 数组并调整维度以适应 matplotlib
    #plt.show()
    #exit(0)
    return or_grad



#========下方是短轴长度损失和法线损失
import open3d as o3d

if opt.flatten_loss_ratio and iteration > 100:
    scales = gaussians.get_scaling
    min_scale, _ = torch.min(scales, dim=1)
    min_scale = torch.clamp(min_scale, 0, 30)
    flatten_loss = torch.abs(min_scale).mean()
    loss += flatten_loss * opt.flatten_loss_ratio  #值设定为100

if opt.normal_loss_ratio and iteration > 3000 and iteration % 500 == 0 and iteration < 15000:
    loss += gaussians.normal_loss() * opt.normal_loss_ratio  #值设定为0.0001

#获取最短轴的向量作为法线
def get_normals(self):
    rotations = self.get_rotation
    rotations_mat = build_rotation(rotations)
    scales = self.get_scaling
    min_scales = torch.argmin(scales, dim=1)
    indices = torch.arange(min_scales.shape[0])
    self.normals = rotations_mat[indices, :, min_scales]
    norms = torch.norm(self.normals, dim=1, keepdim=True)
    norms = torch.clamp(norms, min=1e-8)
    self.normals = self.normals / norms
#削减点的数量，只在被相机包围的范围内且梯度满足要求才计算误差
def cut(self):
    cameras_center = torch.tensor(self.cameras_center, dtype=torch.float, device='cuda')
    radius = self.spatial_lr_scale

    distances = torch.norm(self.get_xyz - cameras_center, dim=1)
    bool_tensor = distances < radius
    grads = self.xyz_gradient_accum / self.denom
    grads[grads.isnan()] = 0.0
    selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= 0.0002, True, False)
    selected_pts_mask = torch.logical_and(selected_pts_mask,
                                          torch.max(self.get_scaling, dim=1).values <= self.percent_dense * radius)
    mask = torch.logical_and(bool_tensor, selected_pts_mask)
    return mask
#计算高斯球法线和邻近几个高斯球的误差
#如果在平面上，那法线应该垂直与点的连线。内积为0，如果法线和表面相切，那误差为1
def normal_loss(self):
    points = self.get_xyz.detach().cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    mask = self.cut()
    self.get_normals()
    dot_product = torch.zeros(1, dtype=torch.float, device='cuda')

    knn_tree = o3d.geometry.KDTreeFlann(pcd)
    k = 5
    knn_indexs_np = [knn_tree.search_knn_vector_3d(p, knn=k)[1] for p in pcd.points]
    knn_indexs = torch.tensor(knn_indexs_np)
    #上面代码的执行时间非常少，约10^-2量级，下方循环，5万个点需要约20秒
    for knn_index, is_allow in zip(knn_indexs, mask):
        if not is_allow:
            continue
        else:
            current_normal = self.normals[knn_index[0]]
            current_position = self.get_xyz[knn_index[0]]

            for idx in [i + 1 for i in range(k - 1)]:
                nest_position = self.get_xyz[knn_index[idx]]
                nest_noraml = nest_position - current_position
                norms = torch.norm(nest_noraml, dim=0, keepdim=True)
                norms = torch.clamp(norms, min=1e-8)
                nest_noraml = nest_noraml / norms
                dot_product += torch.abs(torch.dot(current_normal, nest_noraml))
    return dot_product.mean()