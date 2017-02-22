import os
import cv2
import math
import h5py
import numpy as np


ORI_SHAPE = 600
SHAPE = 384
ORI_FOLDER_PREFIX = '../chair/data/'
TARGET_FOLDER_PREFIX = '../chair_processed/'
LABEL_FOLDER_PREFIX = '../labels/'
def load_and_crop_write(folder, counter, imgCounter,info):
  # target_folder = TARGET_FOLDER_PREFIX + str(counter) + '/'
  # if not os.path.exists(target_folder):
  #   os.makedirs(target_folder)
  for filename in os.listdir(folder):
    img = cv2.imread(os.path.join(folder, filename))
    if img is not None:
      img_processed = cv2.resize(img[(ORI_SHAPE - SHAPE) // 2: (ORI_SHAPE + SHAPE) // 2,
                            (ORI_SHAPE - SHAPE) // 2: (ORI_SHAPE + SHAPE) // 2],
                            (64, 64), interpolation=cv2.INTER_AREA)
      info["data"][imgCounter] = np.transpose(img_processed * 0.00390625 , (2,0,1))
      # with open(LABEL_FOLDER_PREFIX + "train.txt", "a") as label_file:
      #   label_file.write('chair_processed/' + str(counter) + '_' + filename + " " + str(counter) + "\n")
      info["onehot"][imgCounter][counter] = 1
      pIndex = filename.index('p') + 2
      tIndex = filename.index('t') + 1
      info["angles"][imgCounter][0] = math.cos(math.radians(int(filename[pIndex: pIndex + 2])))
      info["angles"][imgCounter][1] = math.sin(math.radians(int(filename[pIndex: pIndex + 2])))
      info["angles"][imgCounter][2] = math.cos(math.radians(int(filename[tIndex: tIndex + 3])))
      info["angles"][imgCounter][3] = math.sin(math.radians(int(filename[tIndex: tIndex + 3])))
      imgCounter = imgCounter + 1
  return imgCounter

def main():
  counter = 0
  imgCounter = 0
  info = {"data"   : np.full((86366, 3, 64, 64), 0, dtype=np.float16),
          "onehot" : np.full((86366, 1393), 0, dtype=np.uint8),
          "angles" : np.full((86366, 4), 0, dtype=np.float16)}
  for folder in os.listdir(ORI_FOLDER_PREFIX):
    origin_folder = ORI_FOLDER_PREFIX + folder + '/renders/'
    if (os.path.isdir(origin_folder)):
      imgCounter = load_and_crop_write(origin_folder, counter, imgCounter, info)
      counter = counter + 1
  with h5py.File(LABEL_FOLDER_PREFIX + 'all_info1.h5', 'w') as f:
    f['data'] = info['data'][:23183]
    f['onehot'] = info['onehot'][:23183]
    f['angles'] = info['angles'][:23183]
  with h5py.File(LABEL_FOLDER_PREFIX + 'all_info2.h5', 'w') as f:
    f['data'] = info['data'][23183:46366]
    f['onehot'] = info['onehot'][23183:46366]
    f['angles'] = info['angles'][23183:46366]
  with h5py.File(LABEL_FOLDER_PREFIX + 'all_info3.h5', 'w') as f:
    f['data'] = info['data'][46366:69549]
    f['onehot'] = info['onehot'][46366:69549]
    f['angles'] = info['angles'][46366:69549]
  with h5py.File(LABEL_FOLDER_PREFIX + 'all_info4.h5', 'w') as f:
    f['data'] = info['data'][69549:86366]
    f['onehot'] = info['onehot'][69549:86366]
    f['angles'] = info['angles'][69549:86366]
  with open(LABEL_FOLDER_PREFIX + 'all_info.txt', 'w') as f:
    f.write('labels/' + 'all_info1.h5' + "\n")
    f.write('labels/' + 'all_info2.h5' + "\n")
    f.write('labels/' + 'all_info3.h5' + "\n")
    f.write('labels/' + 'all_info4.h5')

if __name__ == '__main__':
  main()

