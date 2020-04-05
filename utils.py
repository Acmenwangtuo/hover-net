import json
from openslide import OpenSlide
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scipy.misc
import scipy.io as sio
from scipy.ndimage import filters, measurements
import matplotlib.pyplot as plt
import cv2
from skimage import measure,color
from skimage.morphology import remove_small_objects, watershed


file_json = 'E:\medical\dataset\DS\shanghai\\annotations\\2018-49902003_new.json'
file_wsi = 'E:\medical\dataset\DS\shanghai\images\\2018-49902003 - 2018-10-10 18.02.27.ndpi'
img_save_path = 'E:\medical\dataset\DS\\trian\img\\'
ann_save_path = 'E:\medical\dataset\DS\\trian\\anno\\' 

def openjson(filejson):
    obj = open(filejson)
    a = json.load(obj)
    obj.close()
    return a

def crop(fileWSI,filejson):
    obj = open(filejson)
    slide = OpenSlide(fileWSI)
    w,h = slide.level_dimensions[0]
    anno = json.load(obj)
    cor,hw = get_size(anno)
    tile = np.array(slide.read_region(cor,0, hw))
    return tile

def get_size(json_dict):
    cor = transform_cor(json_dict)[0]
    zui_max = []
    zui_min = []
    for data in cor:
        # print(type(data))
        temp1 = (data.max(axis=0))
        temp2 = (data.min(axis=0))
        zui_max.append(temp1)
        zui_min.append(temp2)
    zui_min = np.array(zui_min)
    zui_max = np.array(zui_max)

    # print(zui.shape)
    # print(min_x)
    min_x = zui_min.min(axis=0)[0]
    min_y = zui_min.min(axis=0)[1]
    max_x = zui_max.max(axis=0)[0]
    max_y = zui_max.max(axis=0)[1]
    # print(min_x,min_y,max_x,max_y)
    h = int(max_y - min_y)+1
    w = int(max_x - min_x)+1
    x = int(min_x)
    y = int(min_y)
    # print(h,w)
    return (x,y),(w,h)


def transform_cor(json_dict):
    ans = []
    X = []
    Y = []
    for data in json_dict:
        res = [];
        list_1 = data['points']
        i = 0;
        while( i <= len(list_1)-2):
            temp = []
            temp.append(list_1[i])
            temp.append(list_1[i+1])
            temp = np.array(temp)
            res.append(temp)
            # print(temp.shape)
            i=i+2;
        res = np.array(res)
        # print(res.shape)
        # print(res[:,0])
        a = np.mean(res[:,0])
        b = np.mean(res[:,1])
        # print(a,b)
        # a = np.mean(res,axis = 0);
        # b = np.mean(res,axis = 0);
        # print(a, b)
        X.append(a)
        Y.append(b)

        # print(res.shape)
        ans.append(res)
        # ans = np.array(ans)
    return ans,X,Y


def transform_cls(json_dict):
    ans = []
    for data in json_dict:
        temp = data['cls']
        if temp == 'Cell':
            ans.append(1)
        else:
            ans.append(2)
    return ans

def remap(list_instance,x,y):
    for data in list_instance:
        for i in range(data.shape[0]):
            data[i][0] = data[i][0] - x
            data[i][1] = data[i][1] - y
    return list_instance

def remap_x_y(X,Y,x,y):
    for i in range(0,len(X)):
        X[i] = X[i] - x
        Y[i] = Y[i] - y
    return X,Y


def getclslabel(list_cls,list_instance,h,w):
    im = np.zeros((h,w),np.uint8)
    cls_size_list = []
    for cell in list_instance:
        vertices = []
        for i in range(cell.shape[0]):
            vertices.append([cell[i][0],cell[i][1]])
        vertices = np.array(vertices)+0.05
        vertices = vertices.astype('int32')
        # vertices = np.array(list(set([tuple(t) for t in vertices])))
        contour = np.zeros((h,w),np.uint8)
        cv2.drawContours(contour,[vertices],0,1,-1)
        cls_size_list.append(contour)
    for cell,clss in zip(cls_size_list,list_cls):
        im[cell > 0] = clss
    return im

def get_centroid(X,Y,list_instance,h,w,x,y):
    X , Y = remap_x_y(x,y,X,Y)
    print(X)
    im_x = np.zeroes((h,w),np.uint8)
    im_y = np.zeroes((h,w),np.unit8)
    cls_size_list = []
    for cell in list_instance:
        vertices = []
        for i in range(cell.shape[0]):
            vertices.append([cell[i][0],cell[i][1]])
        vertices = np.array(vertices)+0.05
        vertices = vertices.astype('int32')
        # vertices = np.array(list(set([tuple(t) for t in vertices])))
        contour = np.zeros((h,w),np.uint8)
        cv2.drawContours(contour,[vertices],0,1,-1)
        cls_size_list.append(contour)
    for cell,c1,c2 in zip(cls_size_list,X,Y):
        im_x[cell > 0] = c1
        im_y[cell > 0] = c2
    
    return im_x,im_y
 
def getinslabel(list_instance,h,w):
    insts_list = []
    idx = 0
    for cell in list_instance:
        vertices = []
        for i in range(cell.shape[0]):
            vertices.append([cell[i][0],cell[i][1]])
        vertices = np.array(vertices)+0.05
        vertices = vertices.astype('int32')
        # vertices = np.array(list(set([tuple(t) for t in vertices])))
        contour = np.zeros((h,w),np.uint8)
        cv2.drawContours(contour,[vertices],0,idx+1,-1)
        insts_list.append(contour)
        idx = idx+1
    insts_size_list = np.array(insts_list)
    insts_size_list = np.sum(insts_size_list, axis=(1 , 2))
    insts_size_list = list(insts_size_list)
    pair_insts_list = zip(insts_list, insts_size_list)
    # sort in z-axis basing on size, larger on top
    pair_insts_list = sorted(pair_insts_list, key=lambda x: x[1])
    insts_list, insts_size_list = zip(*pair_insts_list)
    ann = np.zeros((h,w), np.uint32)
    for idx, inst_map in enumerate(insts_list):
        ann[inst_map > 0] = idx + 1
    return ann


def gen_label(file_json):
    obj = openjson(file_json)
    ans = transform_cls(obj)
    x,y = get_size(obj)[0]
    w,h = get_size(obj)[1]
    ans = transform_cor(obj)
    list_cls = transform_cls(obj)
    list_instance =  remap(ans,x,y)
    im = getclslabel(list_cls,list_instance,h,w)
    img = getinslabel(list_instance,h,w)
    im = im[:,:,np.newaxis]
    img = img[:,:,np.newaxis]
    res = np.concatenate((img,im),axis=2)
    # res = res[0:1000,0:1000,:]
    return res

def gen_mat(file_json):
    obj = openjson(file_json)
    x,y = get_size(obj)[0]
    w,h = get_size(obj)[1]
    ans = transform_cor(obj)[0]
    X = transform_cor(obj)[1]
    Y = transform_cor(obj)[2]
    X,Y = remap_x_y(X,Y,x,y)
    print(X)
    print(Y)
    list_cls = transform_cls(obj)
    list_cls = np.array(list_cls)
    list_cls = list_cls[:,np.newaxis]
    # img_ins = getinslabel(list_instance,h,w)
    X = np.array(X).astype('float32')
    Y = np.array(Y).astype('float32')
    X = X[:,np.newaxis]
    Y = Y[:,np.newaxis]
    centrod = np.concatenate((X,Y),axis=1)
    # print(list_cls.shape)
    # print(centrod.shape)
    sio.savemat("1.mat",{
        'inst_centroid' : centrod,
        'inst_type' : list_cls
    })




if __name__ == "__main__":
    gen_mat(file_json)
    
    # obj = openjson(file_json)
    # ans = transform_cls(obj)
    # x,y = get_size(obj)[0]
    # w,h = get_size(obj)[1]
    # ans = transform_cor(obj)
    # list_cls = transform_cls(obj)
    # list_instance =  remap(ans,x,y)
    # im = getclslabel(list_cls,list_instance,h,w)
    # img = getinslabel(list_instance,h,w)
    # print(np.unique(img))
    # print(np.unique(im))
    # # img[img>0] = 255
    # print(img.shape)
    # # cv2.imshow("mask.jpg",img)
    # # cv2.waitKey(0)
    # plt.imshow(im)
    # plt.show() 
    # print(ans)
    # print(len(ans))

