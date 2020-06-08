import numpy as np
import cv2
def convert(coordinate):
    """
    :param coordinate: format [x1, y1, x2, y2, x3, y3, x4, y4]
    :param with_label: default True
    :return: format [x_c, y_c, w, h, theta, (label)]
    """

    box = np.int0(coordinate)
    box = box.reshape([4, 2])
    rect1 = cv2.minAreaRect(box)

    x, y, w, h, theta = rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], rect1[2]

    return np.array([x,y,w,h,theta], dtype=np.float32)

if __name__ == "__main__":
    print(convert([1,0,0,1,-1,0,0,-1]))
    print(convert([3,1,1,3,-3,-1,-1,-3]))
