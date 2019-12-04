
import os
import numpy as np
from scipy import spatial


def parse_pair_txt(txt_path='', image_root='', file_ext='jpg', image_root_synthetic='', check_exist=True):

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


def parse_list_file(list_file_path, feature_extractor):
    embeddings = list()
    i = 1
    for img_name in list_file_path:
        i += 1
        embedding = extract_feature(feature_extractor, img_name)
        embeddings.append(embedding)

    return np.concatenate(embeddings)


if __name__ == '__main__':

    from evaluation.metric import verification

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    from models.recognition.VGGFace import VGGFace, extract_feature
    feature_extractor = VGGFace()

    # from models.recognition.LightCNN_9 import LightCNN_9, extract_feature
    # feature_extractor = LightCNN_9()

    # from models.recognition.LightCNN_29v2 import LightCNN_29v2, extract_feature
    # feature_extractor = LightCNN_29v2()


    lfw_img_list, is_same = parse_pair_txt(txt_path='../datasets/LFW/pairs.txt', image_root='../datasets/LFW/images')
    embeddings = parse_list_file(lfw_img_list, feature_extractor)
    similarity = calculate_similarity(embeddings)
    verification(is_same, similarity, verbose=True)