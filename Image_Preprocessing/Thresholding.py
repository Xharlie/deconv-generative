import os
import cv2
import math

ORI_SHAPE = 600
SHAPE = 384
ORI_FOLDER_PREFIX = '../chair/data/'
TARGET_FOLDER_PREFIX = '../chair_segmented/'
LABEL_FOLDER_PREFIX = '../labels/'
def load_and_crop_write(folder, counter):
  # target_folder = TARGET_FOLDER_PREFIX + str(counter) + '/'
  # if not os.path.exists(target_folder):
  #   os.makedirs(target_folder)

  for filename in os.listdir(folder):
    img = cv2.imread(os.path.join(folder,filename))
    if img is not None:
      ret, img_threshold = cv2.threshold(img, 254, 255, cv2.THRESH_BINARY)
      img_processed = cv2.resize(img_threshold[(ORI_SHAPE - SHAPE) // 2: (ORI_SHAPE + SHAPE) // 2,
                            (ORI_SHAPE - SHAPE) // 2: (ORI_SHAPE + SHAPE) // 2],
                            (128, 128), interpolation=cv2.INTER_AREA)
      newfile = TARGET_FOLDER_PREFIX + str(counter) + '_' + filename
      cv2.imwrite(newfile, img_processed)
      with open(LABEL_FOLDER_PREFIX + "segm.txt", "a") as label_file:
        label_file.write('chair_segmented/' + str(counter) + '_' + filename + " " + str(counter) + "\n")

def main():
  counter = 0
  for folder in os.listdir(ORI_FOLDER_PREFIX):
    origin_folder = ORI_FOLDER_PREFIX + folder + '/renders/'
    if (os.path.isdir(origin_folder)):
      load_and_crop_write(origin_folder, counter)
      counter = counter + 1

if __name__ == '__main__':
  main()



