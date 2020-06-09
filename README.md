# FCOS_obb
Oriented Bounding box detector modifying FCOS

## 1.Aim

Evaluate the performance of polarmask on DOTA (Dataset for Object Detection in Aerial Images) dataset and compare with existing state-of-the-art models. Extend the research to including some of the ideas from existing models into polar mask to improve the segmentation.

## 2. Related work

ROI transformer is am two-stage detector which is 2nd in oriented object detection, 4th in horizontal detection of DOTA satellite image dataset
RoI transformer’s training process involves a supervised RROI learner is a module that transforms horizontal ROIs to rotation ROIs. This allows the model to have less number of anchor boxes; this is different from models that make use of large number of rotated anchor to achieve the same output. Rotated ROI warping makes use of the orientation learnt to make the features extracted become rotationally invariant. 

SCRDet is a two-stage detector ranking 8th in oriented object detection task, 6th in horizontal detection task of DOTA satellite image dataset. To summarize the model’s modules and architecture, there are three main parts consisting the model: SF-Net, MDA-Net and rotation branch. SF-Net uses C3,C4 layer of ResNet for fusion to balance the semantic information and location information while ignoring other less relevant facts. MDA-Net (multi-dimensional attention network) enhances the object cues and weakens the non-object information through making use of a branch learning the saliency map of input; in this process attention loss is calculated. Rotation Branch makes use of rotation non maximum suppression as post-processing through the calculation of skew IOU computation. 

R3Det +++ (Refined Rotation RetinaNet) is an anchor based single-stage rotation detector it is ranked 7th in oriented object detection task of DOTA satellite image dataset. To summarize the architecture and modules, R3Det involves a refinement stage for not only the detected box but also the feature map. The refinement of feature map involves a 5x1, 1x5 and 1x1 convolution for refinement of detected box. This can be seen as a measure to deal with large aspect ratio, a rotation single-stage detector is used in a refined manner. 
Using rotated anchor in dense scene achieved higher recall rate compared to the use of horizontal anchors, so the model uses a combination strategy of two types of anchors adopted for refinement. 


## 3.Detail

#### 3.1 DOTA dataset

DOTA dataset, which is a satellite image dataset, has 2806 images where the annotations are given as 4 coordinate points of the oriented bounding box and it has 15 categories including plane,ship, basketball court etc. Within these 2806 images there are over 180 K instances which shows that there are a large number of instances in each image. Despite such complexity of the dataset, the state of the art models for the dataset is achieving over 0.8 mAP (RoI Transformer + additional).
<img src= "https://github.com/chanhopark00/FCOS_obb/blob/master/imgs/8.PNG">

#### 3.2 Horizontal vs Oriented Bounding Box

The choice between horizontal and oriented bounding box can be crucial in some applications. For example ,due to its unique shape and nature of image, it is not possible to accurately locate a marine vessel with a horizontal object detector. This is because vessels can not only come in large scale variations but also angle orientation. After RPN outputs regional proposals, non-maximum suppression is generally applied as an important step in reducing the number of candidates and increasing detection efficiency. However for oriented object problems, overlapping horizontal bounding boxes will make it difficult to distinguish vessels crowded near the port. Applying NMS to the screen area can result in missing targets.
<img src= "https://github.com/chanhopark00/FCOS_obb/blob/master/imgs/9.PNG">


#### 3.3 Structure of FCOS

<img src= "https://github.com/chanhopark00/FCOS_obb/blob/master/imgs/2.PNG">
To make the model invariant to scale variation, FCOS makes use of FPN (feature pyramid network) as the backbone of the model.
The model will output 3 types of output: namely, centeredness, class probability and regression. We can think of class probability as a vector with length K and regression as vector with length 4 (top,right,bottom,left). Each position in the multiple feature map will produce these 3 outputs. 
With the four distances predicted, the model will be able to produce rectangular boxes of any ratio; this implies that there is no need of employing anchor boxes which lead to high computational cost. 

The modification I have made is to change the length of the regression vector from 4 to 5; that is to append a target of angle as well.  

Therefore output of model will have shapes as below:

There are 5 feature map levels (due to FPN structure) and the (w_i, h_i) represents the dimension of feature level.

    Class score = List(feature level, N, 15, w_i , h_i)

    Bbox pred = List(feature level, N , 5, w_i, h_i)

    Centeredness = List (feature level, N , w_i, h_i)


#### 3.4 Target of Box Head

<img src= "https://github.com/chanhopark00/FCOS_obb/blob/master/imgs/3.PNG" width="400">
Here the variables with * superscript are the ground truth. Since we want each of the values to be equal to the ground truth, we set the target to be a value which 0 represents equality with the ground truth. With this target, we apply SmoothL1 loss. This offset is consistent with the RoI tranformer’s rotated region proposal network offset.

#### 3.5. Data Pre/post processing

In order to find out the ground truth values for 4 distances and the angle and training, we employ the following method:

#### 3.5.1.  8 points to [4 distances + angle] (preprocssing)

<img src= "https://github.com/chanhopark00/FCOS_obb/blob/master/imgs/10.PNG" width="500">

This is the contour of the bounding box and color representing the angle with respect to the subject pixel position. Clockwise movement along the contour represents angle increasing from 0 to 360; then we make use of the GT angle of the bbox to find the four rays.

<img src= "https://github.com/chanhopark00/FCOS_obb/blob/master/imgs/11.PNG" width="400">  <img src= "https://github.com/chanhopark00/FCOS_obb/blob/master/imgs/12.PNG" width="400">

Here the red dot represents the point of subject, green dots represents the 4 points allows us to infer the perpendicular distances and purple dot being the original center of the oriented box.

#### 3.5.2.  [4 distances + angle] to [x center, y center, width, height, angle] (post processing)

<img src= "https://github.com/chanhopark00/FCOS_obb/blob/master/imgs/1.PNG" width="500">


#### 3.6 Plan of Action

Reimplement results of Baseline, R3 Det ++, FCOS, ROI transformer on DOTA dataset
  
Obtain results of plain polarmask on DOTA for comparison (instance segmentation)
  
Error Analysis of plain polarmask on DOTA  (instance segmentation)
  
Make changes to modules of polarmask for oriented bounding box prediction. 
  
Error analysis and improvements

## 4.Result

#### 4.1 Loss plot

<img src= "https://github.com/chanhopark00/FCOS_obb/blob/master/imgs/4.PNG">


#### 4.2 Visualization

<img src= "https://github.com/chanhopark00/FCOS_obb/blob/master/imgs/5.PNG">
<img src= "https://github.com/chanhopark00/FCOS_obb/blob/master/imgs/6.PNG">
<img src= "https://github.com/chanhopark00/FCOS_obb/blob/master/imgs/7.PNG">

#### 4.3. Metric

| Model | mAP|
|---|---|
| RetinaNet (Baseline)| 0.51 |
| R3Det | 0.66 |
| RoI Transformer | 0.72|
| FCOS-OBB (Mine) | 0.59 |

#### 4.4 Top 1 class detection

|class|mAP|
|---|---|
|plane|0.8644827688468113|
|baseball-diamond|0.6303450416768105|
|bridge|0.2836234662432951|
|ground-track-field|0.4882575648855269|
|small-vehicle|0.6472025055646514|
|large-vehicle|0.5840595934115256|
|ship|0.6647569189809845|
|tennis-court|0.9068834035721455|
|basketball-court|0.6929957009396605|
|storage-tank|0.7713682587649533|
|soccer-ball-field|0.44913695971245626|
|roundabout|0.5449657166644799|
|harbor|0.5146133776276023|
|swimming-pool|0.3366705928715302|
|helicopter|0.3861791785489448|


## 5. Further Improvement

As mentioned above, the major challenges to be solved from this point is to deal with small objects in a cluttered scene and fix the problem of angle regression. Referring to the models discussed before, possible methods for improvements would be as such:

    1. Refinement of predictions based on the ideas of R3 det +++
        Refinement has been proven to be effective by different types of refined object detectors (cascade RCNN, RefineDet)
    2. Pixel attention branch on the feature maps of output of FPN neck based on ideas of SCRDet
        This is shown to be especially effective to deal with images with cluttered instances. 
    3. Weighted regression loss
        Similar to focal loss, we aim to put more emphasis on the angle regression because it is a crucial factor in the prediction of oriented boxes; small change in angle leads to large drop in skew IoU.
        Therefore adding weight to the smooth L1 loss in a way to regularize angle prediction can be an effective way to improve performance.


