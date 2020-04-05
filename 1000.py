import cv2
import numpy as np
import os
file_path = './train/images/'
npy_path = './train/labels/'

img_file = os.listdir(file_path)
npy_file = os.listdir(npy_path)
idx = 1
for img,npy in zip(img_file,npy_file):
    new_img = cv2.imread(file_path+img)
    new_img = new_img[0:1000,0:1000,:]
    new_npy = np.load(npy_path+npy)
    new_npy = new_npy[0:1000,0:1000,:]
    cv2.imwrite(file_path+str(idx)+'.png',new_img)
    np.save(npy_path+str(idx)+'.npy',new_npy)
    idx = idx+1
    print(img,npy,"has finished")
