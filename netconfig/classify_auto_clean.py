# coding=utf-8
from __future__ import division
import numpy as np
import sys, os
import caffe
import time
import matplotlib.pyplot as plt
from collections import OrderedDict

# 设置当前目录
caffe_root = '/home/guangyi/dl/caffe-master/'
project_root = '/home/guangyi/projects/auto-clean/'
verif_path = os.path.join(project_root, 'imgdata/validate')
verif_result = os.path.join(verif_path, 'verif_result.txt')
verif_result_img = os.path.join(verif_path, 'verif_result.jpg')

net_file = project_root + 'netconfig/deploy_auto_clean.prototxt'
caffe_model = project_root + 'netconfig/netmodel/_iter_1000.caffemodel'
mean_file = project_root + 'netconfig/mean.npy'
labels_file = project_root + 'netconfig/words.txt'

temp_time = time.time()
labels = []
labels_dict = {}


# return type_path_list and type_list
def file_list():
    type_path_list = []
    type_list = []
    if os.path.isdir(verif_path):
        type_list = os.listdir(verif_path)
        for type in type_list:
            type_full_path = os.path.join(verif_path, type)
            type_path_list.append(type_full_path)
    return type_path_list, type_list


def init_caffe():
    sys.path.insert(0, caffe_root + 'python')
    # 方法用于改变当前工作目录到指定的路径
    os.chdir(caffe_root)
    net = caffe.Net(net_file, caffe_model, caffe.TEST)
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))
    transformer.set_raw_scale('data', 255)
    transformer.set_channel_swap('data', (2, 1, 0))
    return net, transformer


def init_label():
    global labels
    global labels_dict
    i = 0
    labels = np.loadtxt(labels_file, str, delimiter='\t')
    for label in labels:
        labels_dict[i] = label
        i += 1


def rename_img(p_path, prefix, img, result_str, temp_num):
    _, suffix = os.path.splitext(img)
    short_name = '%s-%05d%d' % (result_str, temp_num, temp_time)
    old = os.path.join(p_path, img)
    new = os.path.join(p_path, '%d-%s%s' % (prefix, short_name, suffix))
    # print(old)
    # print(new)
    os.rename(old, new)


def write_file(class_list, top_1_list, top_3_list, top_5_list, clean_num_list, totall_number_list, time_consume_list):
    result_file = open(verif_result, 'w')
    top_5_dict = {}
    for i in range(0, len(class_list)-1):
        if totall_number_list[i] < 0.001:
            totall_number_list[i] = 0.001
        top_5_dict[i] = (top_5_list[i] / totall_number_list[i] ) * 100
    ordered_dict = OrderedDict(sorted(top_5_dict.items(), key=lambda x: x[1]))
    len_dict = len(ordered_dict)
    top_1_total = 0
    top_3_total = 0
    top_5_total = 0
    clean_total = 0
    for k, v in ordered_dict.items():
        temp = '-------------%s-------------' % (class_list[k])
        print(temp)
        result_file.write(temp + '\n')
        clean_ratio = (clean_num_list[k] / totall_number_list[k]) * 100
        top_1_ratio = (top_1_list[k] / totall_number_list[k]) * 100
        top_3_ratio = (top_3_list[k] / totall_number_list[k]) * 100
        top_1_total += top_1_ratio
        top_3_total += top_3_ratio
        top_5_total += v
        clean_total += clean_ratio
        temp = 'totall_number:%d;   top_1:%.2f%%;   top_3:%.2f%%;  top_5:%.2f%%;   clean_ratio:%.2f%%'% (
            totall_number_list[k], top_1_ratio, top_3_ratio, v, clean_ratio)
        print(temp)
        result_file.write(temp + '\n')
        temp = 'time consume: %.6f, avg: %.6f  (ms)\n' % (time_consume_list[k],
                                                          time_consume_list[k] / totall_number_list[k])
        print(temp)
        result_file.write(temp + '\n')
    temp = 'average:   top_1:%.2f%%;   top_3:%.2f%%;   top_5:%.2f%%;   clean_ratio:%.2f%%'% (
        top_1_total/len_dict, top_3_total/len_dict, top_5_total/len_dict, clean_total/len_dict)
    print(temp)
    result_file.write(temp + '\n')
    # print the result of all-others
    temp = '-------------%s-------------' % (class_list[-1])
    print(temp)
    result_file.write(temp + '\n')
    clean_ratio = clean_num_list[-1] / totall_number_list[-1]
    temp = 'clean ratio: %.2f%%,  time consume: %.6f, avg: %.6f  (ms)\n' % (clean_ratio, time_consume_list[-1],
                                                      time_consume_list[-1] / totall_number_list[-1])
    print(temp)
    result_file.write(temp + '\n')
    result_file.close()
    return


def draw_curve(type_list, top_1_list, top_3_list, top_5_list, totall_number_list):
    # 添加图形属性
    plt.xlabel('%d种食材' % len(type_list))
    plt.ylabel('百分比')
    plt.title('模型的验证结果')
    a = plt.subplot(1, 1, 1)

    x_labels = []
    width = 3
    x = np.arange(10, 75, 10)
    for i in range(0, len(type_list)):
        x_labels.append('%s/%d' % (type_list[i], totall_number_list[i]))
        top_1_list[i] = top_1_list[i] / totall_number_list[i] * 100
        top_3_list[i] = top_3_list[i] / totall_number_list[i] * 100
        top_5_list[i] = top_5_list[i] / totall_number_list[i] * 100

    plt.bar(x - width, top_1_list, facecolor='red', width=width, label='top 1')
    plt.bar(x, top_3_list, facecolor='green', width=width, label='top 3')
    plt.bar(x + width, top_5_list, facecolor='blue', width=width, label='top 5')
    plt.legend()
    plt.xticks(x, x_labels)
    plt.grid(True)
    plt.savefig(verif_result_img)
    return


def recognise_clean(net, transformer, type_path, cur_type):
    temp_num = 0
    cur_type_number = -1
    img_list = os.listdir(type_path)
    totall_number = len(img_list)
    top_1 = 0
    top_3 = 0
    top_5 = 0
    clean_num = 0

    # 找到这个品类对应的标号
    for i in np.arange(len(labels)):
        if labels[i] == cur_type:
            cur_type_number = i
            break
    if cur_type_number == -1:
        print('未找到相应的种类')
        return
    print('正在识别：%s' % cur_type)
    time_consume = time.time()
    for img in img_list:
        temp_num += 1
        img_full_path = os.path.join(type_path, img)
        img_data = caffe.io.load_image(img_full_path)
        net.blobs['data'].data[...] = transformer.preprocess('data', img_data)
        out = net.forward()
        probe_clean = net.blobs['prob_clean'].data[0].flatten()
        sorted_index_clean = probe_clean.argsort()[-1:-3:-1]
        if cur_type_number < 41 and sorted_index_clean[0] == 1:
            clean_num += 1
        top = net.blobs['prob_type'].data[0].flatten()
        top_k = top.argsort()[-1:-6:-1]
        for i in np.arange(top_k.size):
            if top_k[i] <= len(labels) and top_k[i] == cur_type_number:
                result_str = '%s%.2f-%s%.2f-%s%.2f' % (labels_dict[top_k[0]], top[top_k[0]],
                                                       labels_dict[top_k[1]], top[top_k[1]],
                                                       labels_dict[top_k[2]], top[top_k[2]])
                if i == 0:
                    rename_img(type_path, i + 1, img, result_str, temp_num)
                    top_1 += 1
                elif i == 1:
                    rename_img(type_path, i + 1, img, result_str, temp_num)
                    top_3 += 1
                elif i == 2:
                    rename_img(type_path, i + 1, img, result_str, temp_num)
                    top_3 += 1
                elif i == 3:
                    rename_img(type_path, i + 1, img, result_str, temp_num)
                    top_5 += 1
                elif i == 4:
                    rename_img(type_path, i + 1, img, result_str, temp_num)
                    top_5 += 1
                else:
                    rename_img(type_path, 6, img, result_str, temp_num)
    top_3 += top_1
    top_5 += top_3
    time_consume = (time.time() - time_consume) * 1000
    return totall_number, top_1, top_3, top_5, clean_num, time_consume

def recognise_dirty(net, transformer, type_path, cur_type):
    temp_num = 0
    img_list = os.listdir(type_path)
    totall_number = len(img_list)
    dirty_num = 0

    print('正在识别脏图片：%s' % cur_type)
    time_consume = time.time()
    for img in img_list:
        temp_num += 1
        img_full_path = os.path.join(type_path, img)
        img_data = caffe.io.load_image(img_full_path)
        net.blobs['data'].data[...] = transformer.preprocess('data', img_data)
        out = net.forward()
        probe_clean = net.blobs['prob_clean'].data[0].flatten()
        sorted_index_clean = probe_clean.argsort()[-1:-3:-1]
        if sorted_index_clean[0] == 0:
            dirty_num += 1
    time_consume = (time.time() - time_consume) * 1000
    return totall_number, dirty_num, time_consume

if __name__ == "__main__":
    net, transformer = init_caffe()
    init_label()
    type_path_list, type_list = file_list()
    class_list = []
    top_1_list = []
    top_3_list = []
    top_5_list = []
    clean_num_list = []
    totall_number_list = []
    time_consume_list = []
    clean_num_dirty = 0
    totall_number_dirty = 0
    time_consume_dirty = 0
    for i in np.arange(len(type_list)):
        if os.path.isdir(type_path_list[i]) and type_list[i] != 'all-others':
            totall_number, top_1, top_3, top_5, clean_num, time_consume = \
                recognise_clean(net, transformer, type_path_list[i], type_list[i])
            top_1_list.append(top_1)
            top_3_list.append(top_3)
            top_5_list.append(top_5)
            class_list.append(type_list[i])
            clean_num_list.append(clean_num)
            totall_number_list.append(totall_number)
            time_consume_list.append(time_consume)
        elif os.path.isdir(type_path_list[i]) and type_list[i] == 'all-others':
            totall_number_dirty, clean_num_dirty, time_consume_dirty = \
                recognise_dirty(net, transformer, type_path_list[i], type_list[i])
    class_list.append('all-other')
    clean_num_list.append(clean_num_dirty)
    totall_number_list.append(totall_number_dirty)
    time_consume_list.append(time_consume_dirty)

    write_file(class_list, top_1_list, top_3_list, top_5_list, clean_num_list, totall_number_list, time_consume_list)
    # draw_curve(class_list, top_1_list, top_3_list, top_5_list, totall_number_list)
