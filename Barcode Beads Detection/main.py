import cv2
import argparse
import os
import time
import func
import config

tStart = time.time()
#%% args
parser = argparse.ArgumentParser()
parser.add_argument('-I','--image',
                   default='./img/W_A1_0_3.jpg',
                   help='input image')

parser.add_argument('-O','--output',
                    default='./result',
                    help='output path')

args = parser.parse_args()

#%% read & gray
image_org = cv2.imread(args.image)
image = func._rgb2gray(image_org)
#%% conv2
img_cv = func._conv2d(image, config.k_cv)
#%% adpt_thold
img_adpt_thold = func._adpt_thold(img_cv, kernel_size=config.k_adpt_size, c=config.c)
#%% dilation
img_d = func._dilation(img_adpt_thold, config.k_d)
#%% erosion
img_e = func._erosion(img_d, config.k_e)
#%% save
result = (~img_e.astype(bool)) * 255
fn = args.image.split('/')[-1]
cv2.imwrite(os.path.join(args.output, 'result_' + fn), result)

tEnd = time.time()
print ("\n" + "It cost {:.4f} sec" .format(tEnd-tStart))