import torch
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mmdet.core.bbox import RotBox2Polys_torch
class rot_AnchorGenerator(object):

    def __init__(self, scales, ratios, angles, scale_major=True, ctr=None):
        #self.base_size = base_size
        self.scales = torch.Tensor(scales)
        self.ratios = torch.Tensor(ratios)
        self.angles = angles
        self.scale_major = scale_major
        self.ctr = ctr

        # self.base_anchors = self.gen_base_anchors(1,1,1)


    def gen_base_anchors(self,base_size,feat_h, feat_w, stride):
        '''
        :param self.base_size:
        :param self.anchor_scales:
        :param self.ratios:
        :param anchor_thetas:
        :param featuremap_height:
        :param featuremap_width:
        :param stride:
        :return:
        ''' 
        base_anchor = np.array([0, 0, base_size, base_size]).astype(np.float)  # [y_center, x_center, h, w]
        ws, hs, angles = self.enum_ratios_and_thetas(self.enum_scales(base_anchor, self.scales),
                                                self.ratios, self.angles)  # per locations ws and hs and thetas
        x_centers = np.arange(feat_w).astype(np.float) * stride + stride // 2
        y_centers = np.arange(feat_h).astype(np.float) * stride + stride // 2

        x_centers, y_centers = np.meshgrid(x_centers, y_centers)
        angles, _ = np.meshgrid(angles, x_centers)
        ws, x_centers = np.meshgrid(ws, x_centers)
        hs, y_centers = np.meshgrid(hs, y_centers)

        anchor_centers = np.stack([x_centers, y_centers], 2)
        anchor_centers = anchor_centers.reshape([-1, 2])

        box_parameters = np.stack([ws, hs, angles], 2)
        box_parameters = box_parameters.reshape([-1, 3])
        anchors = np.concatenate([anchor_centers, box_parameters], 1)

        return torch.Tensor(anchors).cuda()

    def enum_scales(self,base_anchor, anchor_scales):
        anchor_scales = base_anchor * np.array(anchor_scales).astype(np.float).reshape((len(anchor_scales), 1))

        return anchor_scales


    def enum_ratios_and_thetas(self,anchors, anchor_ratios, anchor_angles):
        '''
        ratio = h /w
        :param anchors:
        :param anchor_ratios:
        :return:
        '''
        ws = anchors[:, 2]  # for base anchor: w == h
        hs = anchors[:, 3]
        anchor_angles = np.array(anchor_angles).astype(np.float)
        sqrt_ratios = np.sqrt(np.array(anchor_ratios))

        ws = (ws / sqrt_ratios[:, np.newaxis]).reshape(-1)
        hs = (hs * sqrt_ratios[:, np.newaxis]).reshape(-1)

        ws, _ = np.meshgrid(ws, anchor_angles)
        hs, anchor_angles = np.meshgrid(hs, anchor_angles)

        anchor_angles = anchor_angles.reshape(-1, 1)
        ws = ws.reshape(-1, 1)
        hs = hs.reshape(-1, 1)

        return ws, hs, anchor_angles


    '''
    def grid_anchors(self, featmap_size, stride=16, device='cuda'):
        base_anchors = self.base_anchors.to(device)

        feat_h, feat_w = featmap_size
        shift_x = torch.arange(0, feat_w, device=device) * stride
        shift_y = torch.arange(0, feat_h, device=device) * stride
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        zeros = torch.zeros_like(shift_xx)
        shifts = torch.stack([shift_xx, shift_yy, zeros, zeros,zeros], dim=-1)
        shifts = shifts.type_as(base_anchors)
        # first feat_w elements correspond to the first row of shifts
        # add A anchors (1, A, 4) to K shifts (K, 1, 4) to get
        # shifted anchors (K, A, 4), reshape to (K*A, 4)

        all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
        all_anchors = all_anchors.view(-1, 5)
        # first A rows correspond to A anchors of (0, 0) in feature map,
        # then (0, 1), (0, 2), ...
        return all_anchors

    def valid_flags(self, featmap_size, valid_size, device='cuda'):
        feat_h, feat_w = featmap_size
        valid_h, valid_w = valid_size
        assert valid_h <= feat_h and valid_w <= feat_w
        valid_x = torch.zeros(feat_w, dtype=torch.uint8, device=device)
        valid_y = torch.zeros(feat_h, dtype=torch.uint8, device=device)
        valid_x[:valid_w] = 1
        valid_y[:valid_h] = 1
        valid_xx, valid_yy = self._meshgrid(valid_x, valid_y)
        valid = valid_xx & valid_yy
        valid = valid[:, None].expand(
            valid.size(0), self.num_base_anchors).contiguous().view(-1)
        return valid
    ''' 


if __name__ == "__main__":
    base_anchor_size = 256
    anchor_scales = [1.]    
    anchor_ratios = [0.5, 2.0, 1/3, 3, 1/5, 5, 1/8, 8]
    anchor_angles = [-90, -75, -60, -45, -30, -15]
    gen = rot_AnchorGenerator(base_anchor_size,anchor_scales,anchor_ratios,anchor_angles)
    print(gen.base_anchors.shape)
    print(gen.gen_base_anchors(2,2,4))
    # print(gen.grid_anchors([16,16]).shape)
    # print(gen.valid_flags(gen.grid_anchors([16,16])).shape)
    # anchors = RotBox2Polys_torch(gen.base_anchors)
    # colors = cm.rainbow(np.linspace(0, 1, anchors.shape[0]))

    # print(anchors.shape)
    # for i,color in zip(range(anchors.shape[0]),colors):
    #     plt.scatter(anchors[i,::2], anchors[i,1::2])
    #     plt.title(str([anchors[i,::2],anchors[i,1::2]])) 
    #     plt.savefig('./anchors/generated_Rot_anchors_%i.png'%(i))       
    #     plt.clf()
