import os
import time
import shutil
import threading
import sys

dataset_root = "/home/fla/workspace/l5kit_data/rasterized"
# dataset_ids = [1,2,3,4]
dataset_ids = [sys.argv[1]]

destination_root = "/fulian_data/rasterized"

start_time = time.time()

for dataset_id in dataset_ids:

	print("Start: ",time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
	target_path = os.path.join(dataset_root, "target_{}".format(dataset_id))
	image_path = os.path.join(dataset_root, "image_{}".format(dataset_id))
	index_path = os.path.join(dataset_root, "index_{}.txt".format(dataset_id))
	destination_image_path = os.path.join(destination_root, "image_{}".format(dataset_id))
	destination_target_path = os.path.join(destination_root, "target_{}".format(dataset_id))

	print("[{}] Reading Index {} ".format(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()), index_path))
	with open(index_path, 'r') as f:
		index_list = [index.strip() for index in f.readlines()]

	if not os.path.exists(destination_image_path):
		os.makedirs(destination_image_path)
	if not os.path.exists(destination_target_path):
		os.makedirs(destination_target_path)
	
	def copy_files_single(index_list, thread_id):
		num_items = len(index_list)
		for i, index in enumerate(index_list, start=1):
			src_image = os.path.join(image_path, index+'.png')
			src_target = os.path.join(target_path, index+'.txt')
			new_image = os.path.join(destination_image_path, index+'.png')
			new_target = os.path.join(destination_target_path, index+'.png')
			shutil.copyfile(src_image, new_image)
			shutil.copyfile(src_target, new_target)
			used_time = time.time() - start_time
			esti_time = (used_time*(num_items-i)/i)/3600
			if i%1000==0:
				print("Thread {} | Dataset_id {} | Item {}/{} | ETA {:.2f} h".format(thread_id, dataset_id, i, num_items, esti_time))
	
	threads_num = 10
	step = int(len(index_list) / threads_num)
	
	print("[{}] Start Processing with {} Threads".format(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()),threads_num))

	thread_list = []
	for thread_id in range(threads_num):
		if not thread_id == threads_num-1:
			index_list_single = index_list[thread_id*step:(thread_id+1)*step]
		else:
			index_list_single = index_list[(threads_num-1)*step:]
		thread = threading.Thread(target=copy_files_single, args=(index_list_single, thread_id))
		thread.start()
		thread_list.append(thread)
	
	for thread in thread_list:
		thread.join()
