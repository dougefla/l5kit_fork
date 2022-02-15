from tkinter.tix import Tree
import numpy as np
from torch import tensor
from torch.utils.data import Dataset
import os
import cv2
import h5py
import random
import time

# class ProcessedDataset(Dataset):
#     def __init__(self, h5_dir, index_dir, history_num_frames, future_num_frames,shuffle:bool):
#         super().__init__()
#         self.h5_file = h5py.File(h5_dir,'r')
#         self.index = index_dir
#         with open(self.index, 'r') as f:
#             self.indexes = f.readlines()
#         self.length = len(self.indexes)
#         self.history_num_frames = history_num_frames
#         self.future_num_frames = future_num_frames

#         self.pool_size = 10000
#         self.pool_start = 0
#         self.pool_end = -1
#         self.image_pool,self.target_pool = self.get_data(self)

#         self.shuffle = shuffle

#     def get_data(self):
#         self.pool_start = self.pool_end+1
#         self.pool_end = self.pool_start+self.pool_size-1
#         image_list = self.h5_file['image'][self.pool_start:self.pool_end+1]
#         target_list = self.h5_file['target'][self.pool_start:self.pool_end+1]
#         if self.shuffle == True:
#             random.seed(0)
#             random.shuffle(image_list)
#             random.seed(0)
#             random.shuffle(target_list)

#         return image_list, target_list

#     def __getitem__(self, index):
        
#         name = (self.indexes[index]).strip()
#         image_path = os.path.join(self.image_dir, name+'.png')
#         target_path = os.path.join(self.target_dir,name+'.txt')
#         image = np.float32(cv2.imread(image_path))
#         if not (os.path.exists(target_path) and os.path.exists(image_path)):
#             return [-1,-1,-1]
#         with open(target_path,'r') as f:
#             target = f.readlines()
#             target_pos = np.float32([eval(i) for i in target[0].split(' ')])
#             # Mistake in pos, only aval for x,y exists in the file
#             # So here we have to extend the aval to x,y,yaw
#             tmp = target[1].split(' ')
#             target_aval = []
#             for i in range(0,len(tmp),2):
#                 aval = eval(tmp[i])
#                 target_aval+=[aval,aval,aval]
#         data = [image, target_pos, np.int8(target_aval)]
#         return data
    
#     def __len__(self):
#         return self.length
    
#     def get_size(self):
#         return self.__len__()


class ProcessedDataset(Dataset):
    def __init__(self, image_dir,target_dir,index_list,history_num_frames,future_num_frames):
        super().__init__()
        self.image_dir = image_dir
        self.target_dir = target_dir
        self.length = len(index_list)
        self.image_list = np.zeros((self.length, 224,224,3))
        self.target_list = np.zeros((self.length, 150))
        self.target_aval_list = np.zeros((self.length, 150))
        self.history_num_frames = history_num_frames
        self.future_num_frames = future_num_frames

    def __getitem__(self, index):
        
        return [self.image_list[index,:,:,:], self.target_list[index,:],self.target_aval_list[index,:]]
    
    def __len__(self):
        return self.length
    
    def get_size(self):
        return self.__len__()
    
    def reload(self, index_list):
        start_time = time.time()
        print("Start Loading From {} to {}".format(index_list[0],index_list[-1]))
        for i, index in enumerate(index_list):
            src_image = os.path.join(self.image_dir, index+'.png')
            src_target = os.path.join(self.target_dir, index+'.txt')
            with open(src_target,'r') as f:
                target = f.readlines()
                target_pos = np.float32([eval(i) for i in target[0].split(' ')])
                # Mistake in pos, only aval for x,y exists in the file
                # So here we have to extend the aval to x,y,yaw
                tmp = target[1].split(' ')
                target_aval = []
                for j in range(0,len(tmp),2):
                    aval = eval(tmp[j])
                    target_aval+=[aval,aval,aval]
            self.image_list[i,:,:,:] = np.float32(cv2.imread(src_image))
            self.target_list[i,:] = target_pos
            self.target_aval_list[i,:] = np.int8(target_aval)
        print("Loaded. Used time: {:.2f} s".format(time.time() - start_time))