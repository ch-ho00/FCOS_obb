import dota_utils as util
import torch
import os
import cv2
import json
import numpy as np 
from PIL import Image
from mmdet.core import get_best_begin_point,polygonToRotRectangle_batch, gt_mask_bp_obbs_list
from math import pi
import pycocotools._mask as _mask
import matplotlib.pyplot as plt
from matplotlib.pyplot import text
import string
import random
import math
wordname_15 = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
               'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter']

wordname_16 = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
               'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter', 'container-crane']

def randomString(stringLength=8):

    letters = string.ascii_lowercase

    return ''.join(random.choice(letters) for i in range(stringLength))


def RotBox2Polys(dboxes):
    """

    :param dboxes:
    :return:
    """
    cs = torch.cos(dboxes[:, 4])
    ss = torch.sin(dboxes[:, 4])
    w = dboxes[:, 2]
    h = dboxes[:, 3]

    x_ctr = dboxes[:, 0]
    y_ctr = dboxes[:, 1]
    x1 = x_ctr + cs * (w / 2.0) - ss * (-h / 2.0)
    x2 = x_ctr + cs * (w / 2.0) - ss * (h / 2.0)
    x3 = x_ctr + cs * (-w / 2.0) - ss * (h / 2.0)
    x4 = x_ctr + cs * (-w / 2.0) - ss * (-h / 2.0)

    y1 = y_ctr + ss * (w / 2.0) + cs * (-h / 2.0)
    y2 = y_ctr + ss * (w / 2.0) + cs * (h / 2.0)
    y3 = y_ctr + ss * (-w / 2.0) + cs * (h / 2.0)
    y4 = y_ctr + ss * (-w / 2.0) + cs * (-h / 2.0)

    polys = torch.cat((x1.unsqueeze(1),
                    y1.unsqueeze(1),
                    x2.unsqueeze(1),
                    y2.unsqueeze(1),
                    x3.unsqueeze(1),
                    y3.unsqueeze(1),
                    x4.unsqueeze(1),
                    y4.unsqueeze(1)), 1)

    return polys

def get_best_begin_point_single(coordinate):
 #   print("????????????",coordinate,flush=True)
    x1 = coordinate[0][0]
    y1 = coordinate[0][1]
    x2 = coordinate[1][0]
    y2 = coordinate[1][1]
    x3 = coordinate[2][0]
    y3 = coordinate[2][1]
    x4 = coordinate[3][0]
    y4 = coordinate[3][1]
    xmin = min(x1, x2, x3, x4)
    ymin = min(y1, y2, y3, y4)
    xmax = max(x1, x2, x3, x4)
    ymax = max(y1, y2, y3, y4)
    combinate = [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]], [[x2, y2], [x3, y3], [x4, y4], [x1, y1]],
                 [[x3, y3], [x4, y4], [x1, y1], [x2, y2]], [[x4, y4], [x1, y1], [x2, y2], [x3, y3]]]
    dst_coordinate = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
    force = 100000000.0
    force_flag = 0
    for i in range(4):
        temp_force = cal_line_length(combinate[i][0], dst_coordinate[0]) + cal_line_length(combinate[i][1],
                                                                                           dst_coordinate[
                                                                                               1]) + cal_line_length(
            combinate[i][2], dst_coordinate[2]) + cal_line_length(combinate[i][3], dst_coordinate[3])
        if temp_force < force:
            force = temp_force
            force_flag = i
    if force_flag != 0:
        pass
        # print("choose one direction!")
    return  combinate[force_flag]

def get_best_begin_point_warp_single(coordinate):

    return TuplePoly2Poly(get_best_begin_point_single(coordinate))

def get_best_begin_point(coordinate_list):
    best_coordinate_list = map(get_best_begin_point_warp_single, coordinate_list)
    # import pdb
    # pdb.set_trace()
    best_coordinate_list = np.stack(list(best_coordinate_list))
    return best_coordinate_list


def cal_line_length(point1, point2):
    return math.sqrt( math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2))

def TuplePoly2Poly(poly):
    outpoly = [poly[0][0], poly[0][1],
                       poly[1][0], poly[1][1],
                       poly[2][0], poly[2][1],
                       poly[3][0], poly[3][1]
                       ]
    return outpoly

def backward_convert(coordinate):
    """
    :param coordinate: format [x1, y1, x2, y2, x3, y3, x4, y4]
    :param with_label: default True
    :return: format [x_c, y_c, w, h, theta, (label)]
    """

    box = np.int0(coordinate)
    box = box.reshape([4, 2])
    rect1 = cv2.minAreaRect(box)

    x, y, w, h, theta = rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], rect1[2]
    before = [x,y,w,h,theta]

    if theta == -90:
        theta = 90
    elif theta == 0:
        tmp = w
        w = h
        h = tmp 
    else:
       if w < h:
           tmp = w
           w = h
           h = tmp
           theta = 90 + theta
    '''
    after = [x,y,w,h,theta]
    plt.title(str([int(a) for a in after]))
    plt.figtext(0.1,0.9, str([int(a) for a in before]))
    plt.plot([x[0] for x in box], [x[1] for x in box])
    plt.axis('scaled')
    plt.savefig('./test/'+randomString(4)+".png")
    plt.clf()
    '''    
    return [x,y,w,h,theta]

def DOTA2COCOTrain(srcpath, destfile, cls_names, difficult='2'):
    # set difficult to filter '2', '1', or do not filter, set '-1'

    imageparent = os.path.join(srcpath, 'images')
    labelparent = os.path.join(srcpath, 'labelTxt')

    data_dict = {}
    data_dict['images'] = []
    data_dict['categories'] = []
    data_dict['annotations'] = []
    for idex, name in enumerate(cls_names):
        single_cat = {'id': idex + 1, 'name': name, 'supercategory': name}
        data_dict['categories'].append(single_cat)

    inst_count = 1
    image_id = 1
    with open(destfile, 'w') as f_out:
        filenames = util.GetFileFromThisRootDir(labelparent)
        for i,file in enumerate(filenames):
            basename = util.custombasename(file)
            # image_id = int(basename[1:])

            imagepath = os.path.join(imageparent, basename + '.png')
            img = cv2.imread(imagepath)
            height, width, c = img.shape

            single_image = {}
            single_image['file_name'] = basename + '.png'
            single_image['id'] = image_id
            single_image['width'] = width
            single_image['height'] = height
            data_dict['images'].append(single_image)

            # annotations
            objects = util.parse_dota_poly2(file)
            for obj in objects:
                if obj['difficult'] == difficult:
                    print('difficult: ', difficult)
                    continue
                single_obj = {}
                single_obj['area'] = obj['area']
                single_obj['category_id'] = cls_names.index(obj['name']) + 1
                single_obj['segmentation'] = []
                single_obj['segmentation'].append(obj['poly'])
                single_obj['iscrowd'] = 0
                xmin, ymin, xmax, ymax = min(obj['poly'][0::2]), min(obj['poly'][1::2]), \
                                         max(obj['poly'][0::2]), max(obj['poly'][1::2])

                w = xmax - xmin
                h = ymax - ymin
                single_obj['hbbox'] = [(xmin + xmax)//2, (ymin + ymax)//2, w, h , 0]
                single_obj['image_id'] = image_id
                [coords] = get_best_begin_point([list(zip(obj['poly'][0::2], obj['poly'][1::2]))])
#                print("before\n",list(zip(obj['poly'][0::2], obj['poly'][1::2])), 'after\n',coords,flush=True)
                gt_obb2 = backward_convert(coords)
                gt_obb2[4] = gt_obb2[4] * np.pi / 180
                [re_poly] = RotBox2Polys(torch.Tensor([gt_obb2])).cpu().numpy()
                '''
                poly =  [[i,j] for i,j in zip(re_poly[0::2], re_poly[1::2])] 
                plt.title(str([int(a) for a in gt_obb2]))
                plt.imshow(img)
                plt.plot([x[0] for x in poly], [x[1] for x in poly],color='red')
                plt.axis('scaled')
                plt.savefig('./test_gt/'+randomString(4)+".png")
                plt.clf()
                ''' 
                single_obj['bbox'] = list(gt_obb2) 
                data_dict['annotations'].append(single_obj)
                single_obj['id'] = inst_count
                inst_count = inst_count + 1
            image_id = image_id + 1
#            if i == 50:
#                break
        json.dump(data_dict, f_out)

def DOTA2COCOTest(srcpath, destfile, cls_names):
    imageparent = os.path.join(srcpath, 'images')
    data_dict = {}

    data_dict['images'] = []
    data_dict['categories'] = []
    for idex, name in enumerate(cls_names):
        single_cat = {'id': idex + 1, 'name': name, 'supercategory': name}
        data_dict['categories'].append(single_cat)

    image_id = 1
    with open(destfile, 'w') as f_out:
        filenames = util.GetFileFromThisRootDir(imageparent)
        for file in filenames:
            basename = util.custombasename(file)
            imagepath = os.path.join(imageparent, basename + '.png')
            img = Image.open(imagepath)
            height = img.height
            width = img.width

            single_image = {}
            single_image['file_name'] = basename + '.png'
            single_image['id'] = image_id
            single_image['width'] = width
            single_image['height'] = height
            data_dict['images'].append(single_image)

            image_id = image_id + 1
        json.dump(data_dict, f_out)

if __name__ == '__main__':


    DOTA2COCOTrain(r'/mnt/lustre/parkchanho/equivariance-project2/roi_transformer/data/dota1_1024/trainval1024/',
                   r'/mnt/lustre/parkchanho/equivariance-project2/roi_transformer/data/dota1_1024/trainval1024/ODAI_train_w_hbb.json',
                   wordname_15)

#    DOTA2COCOTrain(r'/home/dj/code/mmdetection_DOTA/data/dota1_1024_v2/trainval1024_ms',
#                   r'/home/dj/code/mmdetection_DOTA/data/dota1_1024_v2/trainval1024_ms/DOTA_trainval1024_ms.json',
#                   wordname_15)
#    DOTA2COCOTest(r'/home/dj/code/mmdetection_DOTA/data/dota1_1024_v2/test1024',
#                  r'/home/dj/code/mmdetection_DOTA/data/dota1_1024_v2/test1024/DOTA_test1024.json',
#                  wordname_15)
#    DOTA2COCOTest(r'/home/dj/code/mmdetection_DOTA/data/dota1_1024_v2/test1024_ms',
#                  r'/home/dj/code/mmdetection_DOTA/data/dota1_1024_v2/test1024_ms/DOTA_test1024_ms.json',
#                  wordname_15)
