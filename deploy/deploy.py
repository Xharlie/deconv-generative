import numpy as np
import sys
import caffe
import math
import PIL
import cv2

CAFFE_ROOT = '/Users/xharlie/caffe/build/tools/caffe'
ORI_SHAPE = 600
SHAPE = 384

def deploy():
  caffe.set_mode_cpu()
  net = caffe.Net('deconv_deploy.prototxt', 'snapshop_iter_1400000.caffemodel', caffe.TRAIN)
  onehot = np.full((1, 1393), 0, dtype=np.uint8)
  angles = np.full((1, 4), 0, dtype=np.float16)

  onehot[0][400] = 1
  angles[0][0] = math.cos(math.radians(20))
  angles[0][1] = math.sin(math.radians(20))
  angles[0][2] = math.cos(math.radians(220))
  angles[0][3] = math.sin(math.radians(220))
  net.blobs['onehot'].data[...] = onehot
  net.blobs['angles'].data[...] = angles

  resultImage = net.forward()
  print resultImage["deconv9"].shape
  result = resultImage["deconv9"][0] * 256
  # result = np.swapaxes(np.swapaxes(result,0,2),0,1)
  # result = np.transpose(result,(1,2,0))
  result = result.reshape((64,64,3))
  print result.shape
  img = PIL.Image.fromarray(result, 'RGB')
  img.show()

  imgStandard = cv2.imread('/Users/xharlie/dev/deconv-generative/chair/data/1028b32dc1873c2afe26a3ac360dbd4/renders/image_019_p020_t220_r096.png')
  img_processed = cv2.resize(imgStandard[(ORI_SHAPE - SHAPE) // 2: (ORI_SHAPE + SHAPE) // 2,
                             (ORI_SHAPE - SHAPE) // 2: (ORI_SHAPE + SHAPE) // 2],
                             (64, 64), interpolation=cv2.INTER_AREA)
  print img_processed.shape
if __name__ == '__main__':
  deploy()

