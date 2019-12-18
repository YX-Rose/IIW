import os
import numpy as np

from evaluation.metric import rank_accuracy, verification, full_comparison, rank_accuracy_score_fuse
from network.my_public import *
from evaluation.recognition.Select_model import select_model


def parse_line(input_line):
    image_name, image_id = input_line.strip().split(' ')
    return image_name, image_id


def parse_list_file(extract_feature_import, list_file_path, data_root, feature_extractor, strip_NIR=True, isemb = '0'):
    embeddings = list()
    ids = list()
    with open(list_file_path, 'r') as f:
        for line in f.readlines():
            image_name, image_id = parse_line(line)
            image_path = os.path.join(data_root, image_name)

            # if strip_NIR:
                # image_path = image_path.replace('/NIR/', '/')
            if isemb=='0':
                embedding = extract_feature_import(feature_extractor, image_path) #ndarray (1,256)
            else:
                embedding = extract_feature_myModel(feature_extractor, image_path)  # ndarray (1,256)
            embeddings.append(embedding)
            ids.append(image_id)
    return embeddings, ids


def evaluate(isemb, base_root, probe_root, gallery_root, list_root,
             probe_root_original="../../datasets/M2FPA/collected_fr_data_pitch_align/", weight=1.0, testlist=[],
             rec_model_epoch='7', rec_model='', score_fuse=''):
    print('weight = ' + str(weight))

    feature_extractor, extract_feature_import = select_model(rec_model)

    my_model_path = base_root + 'checkpoint'
    feature_extractor_my = myModel(model_path=my_model_path, rec_model_epoch=rec_model_epoch)

    rank_accs = list()
    tpr1s = list()
    tpr01s = list()
    tpr001s = list()

    # testlist = [15, 30, 45, 60, 75, 90]
    resultList = []
    for index in range(len(testlist)):
        print("the angle is " + testlist[index])

        probe_list_path = os.path.join(list_root, 'probe_' + str(testlist[index]) + '.txt')
        gallery_list_path = os.path.join(list_root, 'gallery0' + '.txt')

        if isemb=='0':
            probe_embeddings, probe_ids = parse_list_file(extract_feature_import, probe_list_path, probe_root, feature_extractor)
            gallery_embeddings, gallery_ids = parse_list_file(extract_feature_import, gallery_list_path, gallery_root,
                                                              feature_extractor)  ## groundtruth 051-06 face
        else:
            probe_embeddings, probe_ids = parse_list_file('',probe_list_path, probe_root_original, feature_extractor_my, isemb=isemb) ## generated
            gallery_embeddings, gallery_ids = parse_list_file('', gallery_list_path, gallery_root,
                                                              feature_extractor_my, isemb=isemb)  ## groundtruth 051-06 face
        # gallery_embeddings, gallery_ids = parse_list_file(gallery_list_path, gallery_root, feature_extractor) ## groundtruth 051-06 face

        # if weight != 1:
        probe_embeddings_original, _ = parse_list_file(
            extract_feature_import, probe_list_path, probe_root_original, feature_extractor, strip_NIR=False)
        gallery_embeddings_original, _ = parse_list_file(
            extract_feature_import, gallery_list_path, gallery_root, feature_extractor)

        if score_fuse == '0':
            probe_embeddings = [weight * item_1 + (1 - weight) * probe_embeddings_original[idx_1] for idx_1, item_1 in enumerate(probe_embeddings)]
            gallery_embeddings = [weight * item_2 + (1 - weight) * gallery_embeddings_original[idx_2] for idx_2, item_2 in enumerate(gallery_embeddings)]

            [rank_acc] = rank_accuracy(gallery_embeddings, gallery_ids, probe_embeddings, probe_ids, ranks=1, verbose=False)

            rank_accs.append(rank_acc)
        else:
            [rank_acc] = rank_accuracy_score_fuse(gallery_embeddings_original, gallery_embeddings,
                                                  gallery_ids, probe_embeddings_original, probe_embeddings,
                                                  probe_ids, ranks=1, verbose=False, weight=weight)
            rank_accs.append(rank_acc)

        printText = '{:.2f}% +- {:.2f}%'.format(np.mean(rank_accs), np.var(rank_accs))

        print('rank-1 accuracy: ' + printText)
        resultList.append(printText)
    return resultList



def save_data(file_save_path, data_save):
    mode = 'a' if os.path.exists(file_save_path) else 'w'
    with open(file_save_path, mode) as f:
        f.write(data_save)

if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    isemb = '1' # 1:rec with embedding 0:rec with generated images
    score_fuse = '1' # 1:score fuse 0:embedding fuse
    weight = 0.5
    print('=' * 50)

    # rec_model = 'VGGFace'
    # rec_model = 'ArcFace'
    # rec_model = 'SphereFace'
    # rec_model = 'MobileFace'
    # rec_model = 'LightCNN_9'
    rec_model = 'LightCNN_29v2'

    testlist = ['+15_0', '+15_15', '+15_30', '+15_45', '+15_60', '+15_75', '+15_90',
                '+30_0', '+30_22', '+30_45', '+30_67', '+30_90',
                '-15_0', '-15_15', '-15_30', '-15_45', '-15_60', '-15_75', '-15_90',
                '-30_0', '-30_22', '-30_45', '-30_67', '-30_90',
                '0_15', '0_30', '0_45', '0_60', '0_75', '0_90']

    # /data1/mandi.luo/work/FaceRotation/cjcode-2-v1/mainBig/model_output/FE_frontalization/MP#
    path_root = "../../pretrained/Ours/MP/"
    gallery_root = "../../datasets/M2FPA/collected_fr_data_pitch_align/"
    list_root = '../../datasets/M2FPA/M2FPA_test_protocol/'

    for i in range(7, 8):
        print("test_model_epoch is " + str(i))
        data = evaluate(isemb=isemb, base_root=path_root, probe_root=path_root + 'output-eval' + str(i) + '/', gallery_root=gallery_root,
                        list_root=list_root, weight=weight, testlist=testlist, rec_model_epoch=str(i), rec_model=rec_model
                        , score_fuse=score_fuse)
        exit()

        for k in range(len(data)):
            data_save = "test_model_epoch " + str(i) + " rank-1 of angle " + testlist[k] + " is " \
                                + str(data[k]) + "\n"
            if isemb=='1':
                file_path = path_root + rec_model + "_M2FPA_emb_Result-" + str(i) + "_weight" + str(weight) + ".txt"
            else:
                file_path = path_root + rec_model + "_M2FPA_Result-" + str(i) + "_weight" + str(weight) + ".txt"
            save_data(file_path, data_save)



