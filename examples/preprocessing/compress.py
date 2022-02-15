import os
import time
import shutil
import threading
import sys
import h5py
import cv2
import numpy as np

resume = False
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
dataset_root = r"/home/fla/workspace/l5kit_data/rasterized"
dataset_ids = [1,2,3,4]
# dataset_ids = [sys.argv[1]]
buffer_size = 100

destination_root = r"/fulian_data/rasterized"

start_time = time.time()

for dataset_id in dataset_ids:

    print("Start: ",time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
    target_path = os.path.join(dataset_root, "target_{}".format(dataset_id))
    image_path = os.path.join(dataset_root, "image_{}".format(dataset_id))
    index_path = os.path.join(dataset_root, "index_{}.txt".format(dataset_id))

    print("[{}] Reading Index {} ".format(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()), index_path))
    with open(index_path, 'r') as f:
        index_list = [index.strip() for index in f.readlines()]
    
    num_items = len(index_list)

    h5f_name = os.path.join(destination_root, 'data_{}.h5'.format(dataset_id))
    if resume and os.path.exists(h5f_name):
        h5f = h5py.File(h5f_name, 'a')
        image_set = h5f['image']
        target_set = h5f['target']
        length = h5f['length']
        # Resume from i=length
        index_list = index_list[length[0]:]
    else:
        h5f = h5py.File(h5f_name, 'w')
        image_set = h5f.create_dataset(name = "image",shape = (num_items, 224, 224, 3), dtype='uint8')
        target_set = h5f.create_dataset(name = "target",shape = (num_items, 150, 2), dtype='int')
        length_set = h5f.create_dataset(name = "length",shape = (1), dtype='int')

    start_time = time.time()

    def load_hdf5(h5f, index_list, offset, thread_id):
        # Use these two list as buffer with size of buffer_size
        image_buffer = np.zeros((buffer_size, 224,224,3))
        target_buffer = np.zeros((buffer_size, 150, 2))
        # counter = 0->buffer_size-1
        num_items = len(index_list)
        counter = 0
        # Enumerate from start, 
        for i, index in enumerate(index_list):
            src_image = os.path.join(image_path, index+'.png')
            src_target = os.path.join(target_path, index+'.txt')
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
            # Load the data from disk to image_buffer and target_buffer
            image_buffer[counter,:,:,:] = cv2.imread(src_image)
            target_buffer[counter,:,0] = target_pos
            target_buffer[counter,:,1] = np.int8(target_aval)
            counter +=1
            # If the buffer is full
            if counter == buffer_size:
                # Load the buffer into the h5 file
                # i = counter-1 + n*buffer_size = buffer_size-1 + n*buffer_size
                h5f['image'][offset+i+1-buffer_size:offset+i+1, :,:,:] = image_buffer
                h5f['target'][offset+i+1-buffer_size:offset+i+1, :,0] = target_buffer[:,:,0]
                h5f['target'][offset+i+1-buffer_size:offset+i+1, :,1] = target_buffer[:,:,1]
                h5f['length'][0] = [i+1]
                # Reset the counter
                counter = 0
                # Close and re-Open to save the file
                # h5f.close()
                # h5f = h5py.File(h5f_name, 'a')
                used_time = time.time() - start_time
                esti_time = (used_time*(num_items-(i+1))/(i+1))/3600
                print("Dataset_id {} | Thread_id {} | Item {}/{} | ETA {:.2f} h".format(dataset_id, thread_id, i+1, num_items, esti_time))

    threads_num = 5
    step = int(len(index_list) / threads_num)
    print("[{}] Start Processing with {} Threads".format(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()),threads_num))

    thread_list = []
    for thread_id in range(threads_num):
        if not thread_id == threads_num-1:
            index_list_single = index_list[thread_id*step:(thread_id+1)*step]
            offset = thread_id*step
        else:
            index_list_single = index_list[(threads_num-1)*step:]
            offset =(threads_num-1)*step
        thread = threading.Thread(target=load_hdf5, args=( h5f, index_list_single,offset,thread_id))
        thread.start()
        thread_list.append(thread)
    for thread in thread_list:
        thread.join()
    h5f.close()