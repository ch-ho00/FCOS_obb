from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
import argparse
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser()
parser.add_argument("--config",nargs='?',const="./configs/polarmask/4gpu/polar_768_1x_r50.py")
parser.add_argument("--checkpoint",nargs='?',const="./checkpoints/r50_1x.pth")
parser.add_argument("--img",nargs='?',const="./data/coco/val2017/000000000139.jpg")
args = parser.parse_args()

config_file = args.config
checkpoint_file = args.checkpoint
model = init_detector(config_file, checkpoint_file, device='cuda:0')
img = args.img 
for scale in [0.3,0.5,0.7,0.9,1]:
    result = inference_detector(model, img,scale)
    show_result_pyplot(img, result, model.CLASSES,out_file='%s_scale_%d'%(config_file[config_file.rfind('/')+1:config_file.rfind('.py')],10*scale),scale=scale)

