
import os
import numpy as np
from scipy import spatial


def parse_pair_txt(txt_path, image_root, file_ext='bmp', image_root_synthetic='', check_exist=True):

    pairs = list()
    for line in open(txt_path, 'r').readlines():
        pair = line.strip().split()
        pairs.append(pair)
    pairs = np.array(pairs)

    if image_root_synthetic == '':
        image_root_synthetic = image_root
    nrof_skipped_pairs = 0

    lfw_img_list = list()
    is_same = list()

    for pair in pairs:
        if len(pair) == 3:
            path0 = os.path.join(image_root, pair[0], pair[0] + '_' + '%04d' % int(pair[1]) + '.' + file_ext)
            path1 = os.path.join(image_root_synthetic, pair[0], pair[0] + '_' + '%04d' % int(pair[2]) + '.' + file_ext)
            issame = True
        elif len(pair) == 4:
            path0 = os.path.join(image_root, pair[0], pair[0] + '_' + '%04d' % int(pair[1]) + '.' + file_ext)
            path1 = os.path.join(image_root_synthetic, pair[2], pair[2] + '_' + '%04d' % int(pair[3]) + '.' + file_ext)
            issame = False

        if check_exist:
            if os.path.exists(path0) and os.path.exists(path1):  # Only add the pair if both paths exist
                lfw_img_list += (path0, path1)
                is_same.append(issame)
            else:
                print(pair)
                nrof_skipped_pairs += 1
        else:
            lfw_img_list += (path0, path1)
            is_same.append(issame)

    if nrof_skipped_pairs > 0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)

    return lfw_img_list, is_same


def calculate_similarity(embeddings):
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    similarity = list()
    for i in range(len(embeddings1)):
        sim = 1 - spatial.distance.cosine(embeddings1[i], embeddings2[i])
        similarity.append(sim)
    return similarity


def parse_list_file(list_file_path, feature_extractor, image_root):
    embeddings = list()
    i = 1
    for img_name in list_file_path:
        i += 1
        image_path = os.path.join(image_root, img_name)

        # if strip_NIR:
        # image_path = image_path.replace('/NIR/', '/')

        embedding = extract_feature_v2(feature_extractor, image_path, 48)
        embeddings.append(embedding)

    return np.concatenate(embeddings)


if __name__ == '__main__':

    from face_editing.evaluation.metric import verification
    # from lightcnn.private import LightCNN29_V4, extract_feature
    # from vgg.recognition import VGGFace, extract_feature

    os.environ['CUDA_VISIBLE_DEVICES'] = '4'

    # from models.recognition.VGGFace import VGGFace, extract_feature
    # feature_extractor = VGGFace(model_path='../../pretrained/VGGFace/model.pth.tar')

    # from models.recognition.LightCNN_9 import LightCNN_9, extract_feature
    # feature_extractor = LightCNN_9()

    from face_editing.lightcnn.public import LightCNN29_v2, extract_feature_v2
    feature_extractor = LightCNN29_v2()

    image_root = '/data1/mandi.luo/dataset/lfw/lfw_align3/lfw/image'
    lfw_img_list, is_same = parse_pair_txt(txt_path=os.path.expanduser('/data1/mandi.luo/dataset/lfw/pairs.txt'),
                                           image_root=os.path.expanduser(image_root))
    embeddings = parse_list_file(lfw_img_list, feature_extractor, image_root)
    similarity = calculate_similarity(embeddings)
    verification(is_same, similarity, verbose=True)
