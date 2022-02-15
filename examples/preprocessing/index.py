# This script check the sub-dataset, and get the index file.
# The missing pairs won't be recorded into the index file, 
# but will be recorded into the missed file.

import os
import time

dataset_root = "/home/fla/workspace/l5kit_data/rasterized"
dataset_ids = [4]

for dataset_id in dataset_ids:

    print("Start: ",time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))

    target_path = os.path.join(dataset_root, "target_{}".format(dataset_id))
    image_path = os.path.join(dataset_root, "image_{}".format(dataset_id))
    index_file_path = os.path.join(dataset_root, "index_{}.txt".format(dataset_id))
    image_missing_files_path = os.path.join(dataset_root, "image_missing_{}.txt".format(dataset_id))
    target_missing_files_path = os.path.join(dataset_root, "target_missing_{}.txt".format(dataset_id))

    print("Start Scanning Folder: {}".format(image_path))
    image_files = sorted(os.listdir(image_path))
    image_files_num = len(image_files)
    print("Scan Successed. Detected {} Items in {}".format(image_files_num, image_path))

    print("Start Scanning Folder: {}".format(target_path))
    target_files = sorted(os.listdir(target_path))
    target_files_num = len(target_files)
    print("Scan Successed. Detected {} Items in {}".format(target_files_num, target_path))

    index_list = []
    image_missing_files = []
    target_missing_files = []

    # i - image_files pointer
    i = 0
    # j - target_files pointer
    j = 0
    # Use dual-pointer to trasverse
    while i < image_files_num and j < target_files_num:
        # Get the basename of both image_file and target_file
        # Which should be like: train_30_50_{:>20d}
        image_file = (image_files[i].split('.'))[0]
        target_file = (target_files[j].split('.'))[0]

        # Check if the file name is legal:
        if not 'train_30_50' in image_file:
            i+=1
            continue
        if not 'train_30_50' in target_file:
            j +=1
            continue

        # Normal Case
        if image_file == target_file:
            index_list.append(image_file+'\n')
            i +=1
            j +=1
        # Image Missing, which should have the same name as target_file
        elif image_file > target_file:
            print("Found target_file But Missing image_file at {}".format(target_file))
            image_missing_files.append(target_file+'\n')
            j +=1
        # Target Missing, which should have the same name as image_file
        elif image_file < target_file:
            print("Found image_file But Missing target_file at {}".format(image_file))
            target_missing_files.append(image_file+'\n')
            i += 1
    # If image_files remained, 
    while i < image_files_num:
        image_file = (image_files[i].split('.'))[0]
        print("Found image_file But Missing target_file at {}".format(image_file))
        target_missing_files.append(image_file+'\n')
        i += 1
    # If target_files remained, 
    while j < target_files_num:
        target_file = (target_files[j].split('.'))[0]
        print("Found target_file But Missing image_file at {}".format(target_file))
        image_missing_files.append(target_file+'\n')
        j += 1
    

    with open(index_file_path, 'w') as f:
        f.writelines(index_list)
        print("Written {} items into {}".format(len(index_list),index_file_path))
    with open(image_missing_files_path, 'w') as f:
        f.writelines(image_missing_files)
        print("Written {} items into {}".format(len(image_missing_files),image_missing_files_path))
    with open(target_missing_files_path, 'w') as f:
        f.writelines(target_missing_files)
        print("Written {} items into {}".format(len(target_missing_files),target_missing_files_path))

    print("End: ",time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))