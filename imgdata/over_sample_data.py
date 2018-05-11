# coding=utf-8
import os
import sys
import shutil

# image_path = 'C:\\Users\\shiqi\\Desktop\\pics'
image_path = sys.argv[1]
train_ratio = 0.7
test_ratio = 0.1
# train_number = 100
# test_number = 20
train_number = sys.argv[2]
test_number = sys.argv[3]
train_foldname = 'train'
test_foldname = 'test'
validat_foldname = 'validate'
words_filename = 'words.txt'
train_filename = 'train.txt'
test_filename = 'test.txt'

words_map = {}


def create_words(words_list):
    words_file = open(words_filename, 'w')
    i = 0
    for food_type in words_list:
        words_file.write(food_type + '\n')
        words_map[food_type] = i
        i += 1
    words_file.close()


def write_filelist(p_path, filename, max_number):
    max_number = int(max_number)
    file = open(filename, 'w')
    food_list = os.listdir(p_path)
    for food_type in food_list:
        food_type_index = words_map[food_type]
        counter = 0
        index = 0
        image_name_list = os.listdir(os.path.join(p_path, food_type))
        while counter < max_number:
            file.write('%s %d\n' % (os.path.join(food_type, image_name_list[index]), food_type_index, ))
            index += 1
            if index >= len(image_name_list):
                index = 0
            counter += 1
        print('%s  count:%d' % (food_type, counter))
    file.close()


def copy_file(all_image_path):
    food_list = os.listdir(all_image_path)
    for food_type in food_list:
        print('is copying %s' % food_type)
        cur_path = os.path.join(all_image_path, food_type)
        image_file_list = os.listdir(cur_path)
        image_number = len(image_file_list)
        if image_number < 10:
            continue
        cur_train_fold = os.path.join(train_foldname, food_type)
        if not os.path.exists(cur_train_fold):
            os.makedirs(cur_train_fold)
        cur_test_fold = os.path.join(test_foldname, food_type)
        if not os.path.exists(cur_test_fold):
            os.makedirs(cur_test_fold)
        cur_validate_fold = os.path.join(validat_foldname, food_type)
        if not os.path.exists(cur_validate_fold):
            os.makedirs(cur_validate_fold)

        train_number = int(image_number * train_ratio)
        test_number = int(image_number * test_ratio)
        validate_number = image_number - train_number - test_number
        index = 0
        counter = 0

        while counter < train_number:
            shutil.copy(os.path.join(cur_path, image_file_list[index]), cur_train_fold)
            index += 1
            counter += 1

        counter = 0
        while counter < test_number:
            shutil.copy(os.path.join(cur_path, image_file_list[index]), cur_test_fold)
            index += 1
            counter += 1

        counter = 0
        while counter < validate_number:
            shutil.copy(os.path.join(cur_path, image_file_list[index]), cur_validate_fold)
            index += 1
            counter += 1


if __name__ == '__main__':
    # prepare data
    if os.path.exists(train_foldname):
        shutil.rmtree(train_foldname)
    if os.path.exists(test_foldname):
        shutil.rmtree(test_foldname)
    if os.path.exists(validat_foldname):
        shutil.rmtree(validat_foldname)
    copy_file(image_path)
    # create file list
    food_list = os.listdir(train_foldname)
    create_words(food_list)
    write_filelist(train_foldname, train_filename, train_number)
    write_filelist(test_foldname, test_filename, test_number)
