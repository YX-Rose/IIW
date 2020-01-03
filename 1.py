from utils import util
import os

input_root = '/data1/xin.ma/datasets/VGGFace2/'
image_path = os.path.join(input_root, 'test')

image_path_rel = []
image_list = sorted(util.make_dataset_list(image_path))
for i in image_list:
   i_rel =  os.path.relpath(i, image_path)
   image_path_rel.append(i_rel)

with open("/data1/xin.ma/datasets/VGGFace2/test.txt", 'w') as f:
    for i in image_path_rel:
        f.writelines(i)
        f.writelines('\n')
