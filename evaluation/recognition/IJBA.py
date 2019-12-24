

import os
import numpy as np
from scipy import spatial


def parse_line_recognition(input_line):
    contents = input_line.strip().split(' ')
    image_name = contents[:-2]
    image_id = contents[-1]
    return image_name, image_id


def parse_line_verification(input_line, template_dict):
    contents = input_line.strip().split(' ')
    template_a = template_dict[contents[0]]
    template_b = template_dict[contents[1]]
    same = True if contents[2] == '1' else False

    return template_a, template_b, same


def parse_protocol_recognition(list_file_path, data_root, feature_extractor):
    embeddings = list()
    ids = list()
    with open(list_file_path, 'r') as f:
        for line in f.readlines():
            image_name, image_id = parse_line_recognition(line)
            image_name = [os.path.join(data_root, item) for item in image_name]

            embedding = extract_template(image_name, feature_extractor)
            embeddings.append(embedding)
            ids.append(image_id)
    return embeddings, ids


def parse_protocol_verification(pair_txt, template_npy, data_root, feature_extractor):

    template_dict = np.load(template_npy)[()]

    embeddings = list()
    is_sames = list()
    with open(pair_txt, 'r') as f:
        for line in f.readlines():
            template_a, template_b, is_same = parse_line_verification(line, template_dict)
            template_a = [os.path.join(data_root, item) for item in template_a]
            template_b = [os.path.join(data_root, item) for item in template_b]

            embedding = extract_template(template_a, feature_extractor)
            embeddings.append(embedding)

            embedding = extract_template(template_b, feature_extractor)
            embeddings.append(embedding)

            is_sames.append(is_same)
    return embeddings, is_sames


def extract_template(image_names, feature_extractor):
    template_len = len(image_names)
    embedding = None
    for image_path in image_names:
        if embedding is None:
            embedding = extract_feature(feature_extractor, image_path)
        else:
            embedding += extract_feature(feature_extractor, image_path)
    embedding = embedding / template_len
    return embedding


def calculate_similarity(embeddings):
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    similarity = list()
    for i in range(len(embeddings1)):
        sim = 1 - spatial.distance.cosine(embeddings1[i], embeddings2[i])
        similarity.append(sim)
    return similarity


def evaluate_recognition(probe_txt, gallery_txt, feature_extractor, data_root="../../datasets/IJB-A/data/", syn_root=''):

    probe_embeddings, probe_ids = parse_protocol_recognition(probe_txt, os.path.expanduser(data_root), feature_extractor=feature_extractor)
    gallery_embeddings, gallery_ids = parse_protocol_recognition(gallery_txt, os.path.expanduser(data_root), feature_extractor=feature_extractor)

    if syn_root != '':
        syn_probe_embeddings, _ = parse_protocol_recognition(probe_txt, os.path.expanduser(syn_root), feature_extractor=feature_extractor)
        syn_gallery_embeddings, _ = parse_protocol_recognition(gallery_txt, os.path.expanduser(syn_root), feature_extractor=feature_extractor)
        probe_embeddings = [(item + syn_probe_embeddings[idx]) / 2 for idx, item in enumerate(probe_embeddings)]
        gallery_embeddings = [(item + syn_gallery_embeddings[idx]) / 2 for idx, item in enumerate(syn_gallery_embeddings)]

    rank_accuracy(gallery_embeddings, gallery_ids, probe_embeddings, probe_ids, ranks=5, verbose=True)


def evaluate_verification(pair_txt, template_npy, feature_extractor, data_root="../../datasets/IJB-A/data/"):

    embeddings, is_same = parse_protocol_verification(pair_txt, template_npy, os.path.expanduser(data_root), feature_extractor)
    similarity = calculate_similarity(embeddings)
    verification(is_same, similarity, verbose=True)


if __name__ == '__main__':

    from evaluation.metric import rank_accuracy, verification
    from models.recognition.LightCNN_29v2 import LightCNN_29v2, extract_feature

    os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    recognizer = LightCNN_29v2(model_path='../../pretrained/LightCNN_29v2/model.pth.tar')

    syn_root = ''

    # evaluate_recognition("../../datasets/IJB-A/protocol/recognition/split2/probe.txt",
    #                      "../../datasets/IJB-A/protocol/recognition/split2/gallery.txt",
    #                      recognizer,
    #                      syn_root=syn_root)

    evaluate_verification('../../datasets/IJB-A/protocol/verification/split2/val_pair.txt',
                          '../../datasets/IJB-A/protocol/verification/split2/val_images.npy',
                          recognizer)

    # for i in range(1, 11):
    #
    #     print('=' * 50)
    #     # print('split : ' + str(i))
    #     print('epoch : ' + str(i))
    #
    #     # evaluate_verification('/home/jie.cao/main/dataset/IJB-A/protocol/verification/split' + str(i) + '/val_pair.txt',
    #     #                       '/home/jie.cao/main/dataset/IJB-A/protocol/verification/split' + str(i) + '/val_images.npy',
    #     #                       recognizer)
    #
    #     evaluate_recognition('/home/jie.cao/main/dataset/IJB-A/protocol/recognition/split2/probe.txt',
    #                          '/home/jie.cao/main/dataset/IJB-A/protocol/recognition/split2/gallery.txt',
    #                          recognizer,
    #                          syn_root='/home/jie.cao/main/model_output/FF_S/V2/IJB-A-' + str(i)
    #                          )
