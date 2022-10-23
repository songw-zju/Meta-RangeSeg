import torch
from torchvision.transforms import functional as FF
import torch.nn.functional as Func
import numpy as np
import time


def NN_filter(proj_range, semantic_pred, k_size=5):
    semantic_pred = semantic_pred.double()
    H, W = np.shape(proj_range)

    proj_range_expand = torch.unsqueeze(proj_range, axis=0)
    proj_range_expand = torch.unsqueeze(proj_range_expand, axis=0)

    semantic_pred_expand = torch.unsqueeze(semantic_pred, axis=0)
    semantic_pred_expand = torch.unsqueeze(semantic_pred_expand, axis=0)

    pad = int((k_size - 1) / 2)

    proj_unfold_range = Func.unfold(proj_range_expand, kernel_size=(k_size, k_size), padding=(pad, pad))
    proj_unfold_range = proj_unfold_range.reshape(-1, k_size * k_size, H, W)

    proj_unfold_pre = Func.unfold(semantic_pred_expand, kernel_size=(k_size, k_size), padding=(pad, pad))
    proj_unfold_pre = proj_unfold_pre.reshape(-1, k_size * k_size, H, W)

    return proj_unfold_range, proj_unfold_pre


def get_semantic_segmentation(sem):
    # map semantic output to labels
    if sem.size(0) != 1:
        raise ValueError('Only supports inference for batch size = 1')
    sem = sem.squeeze(0)
    predict_pre=torch.argmax(sem, dim=0, keepdim=True)
    '''
    sem_prob=Func.softmax(sem,dim=0)
    change_mask_motorcyclist=torch.logical_and(predict_pre==7,sem_prob[8:9,:,:]>0.1)
    predict_pre[change_mask_motorcyclist]=8
    '''
    return predict_pre


inv_label_dict = {0: 0, 1: 10, 2: 11, 3: 15, 4: 18, 5: 20, 6: 30, 7: 31, 8: 32, 9: 40, 10: 44, 11: 48, 12: 49, 13: 50, 14: 51, 15: 70, 16: 71, 17: 72, 18: 80, 19: 81}
device = torch.device('cuda:{}'.format(0))
proj_range = torch.randn(64, 2048)
semantic_output = torch.randn(1, 20, 64, 2048)
semantic_pred = get_semantic_segmentation(semantic_output[:1, :, :, :])
range_img = torch.unsqueeze(FF.to_tensor(proj_range/80.0), axis=0)

t_1 = torch.squeeze(range_img * 80.0).detach().to(device)
t_3 = torch.squeeze(semantic_pred).detach().to(device)

a = time.time()
# proj_unfold_range,proj_unfold_pre=NN_filter(t_1,t_2,t_3,t_4,t_5)
proj_unfold_range, proj_unfold_pre = NN_filter(t_1, t_3)

b = time.time()
# print (b-a)
semantic_pred = np.squeeze(semantic_pred.detach().cpu().numpy())
proj_unfold_range = proj_unfold_range.cpu().numpy()
proj_unfold_pre = proj_unfold_pre.cpu().numpy()
label = []
for jj in range(len(A.proj_x)):
    y_range, x_range = A.proj_y[jj], A.proj_x[jj]
    upper_half = 0
    if A.unproj_range[jj] == proj_range[y_range, x_range]:
        lower_half = inv_label_dict[semantic_pred[y_range, x_range]]
    else:
        potential_label = proj_unfold_pre[0, :, y_range, x_range]
        potential_range = proj_unfold_range[0, :, y_range, x_range]
        min_arg = np.argmin(abs(potential_range - A.unproj_range[jj]))
        lower_half = inv_label_dict[potential_label[min_arg]]
    label_each = (upper_half << 16) + lower_half
    label.append(label_each)