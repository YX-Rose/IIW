import os

root = "/data1/mandi.luo/dataset/M2FPA/M2FPA_test_protocol/gallery0.txt"

img = []
with open(root, 'r') as f:
    for line in f.readlines():
        rel_img = line.strip().split(' ')[0]
        img.append(rel_img)

with open('/data1/mandi.luo/dataset/M2FPA/M2FPA_test_protocol/all_gallery.txt', 'w') as f:
    for i in img:
        f.writelines(i)
        f.writelines('\n')

