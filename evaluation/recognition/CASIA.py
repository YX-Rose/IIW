
import os
import numpy as np

# from face_editing.evaluation.metric import rank_accuracy, verification, full_comparison
# from face_editing.lightcnn.public import LightCNN9, LightCNN29_v2, extract_feature_v2
# from face_editing.lightcnn.public import LightCNN29_V4, extract_feature

from metric import rank_accuracy, verification, full_comparison
from face_editing.lightcnn.public import LightCNN9, LightCNN29_v2, extract_feature_v2




def parse_line(input_line):
    image_name, image_id = input_line.strip().split(' ')
    image_name = image_name.replace('bmp', 'jpg')
    return image_name, image_id


def parse_list_file(list_file_path, data_root, feature_extractor, strip_NIR=True):
    embeddings = list()
    ids = list()
    with open(list_file_path, 'r') as f:
        for line in f.readlines():
            image_name, image_id = parse_line(line)
            image_path = os.path.join(data_root, image_name)

            # if strip_NIR:
                # image_path = image_path.replace('/NIR/', '/')

            embedding = extract_feature_v2(feature_extractor, image_path, 48)
            embeddings.append(embedding)
            ids.append(image_id)
    return embeddings, ids


def evaluate(probe_root, gallery_root, list_root,
             probe_root_original='/data1/mandi.luo/dataset/Multi-PIE/data/', weight=1.0, testlist=[]):
    print('weight = ' + str(weight))

    # feature_extractor = LightCNN9()
    feature_extractor = LightCNN29_v2()
    # feature_extractor = LightCNN29_V4()

    rank_accs = list()
    tpr1s = list()
    tpr01s = list()
    tpr001s = list()

    # testlist = [15, 30, 45, 60, 75, 90]
    resultList = []
    for index in range(len(testlist)):
        print("the angle is " + str(testlist[index]))
        for idx in range(1):
            i = idx + 1
            probe_list_path = os.path.join(list_root, 'multipie_' + str(testlist[index]) + '_test_list.txt')
            # probe_list_path = os.path.join(list_root, 'multipie_gallery_test_list.txt')
            gallery_list_path = os.path.join(list_root, 'multipie_gallery_test_list.txt')

            # probe_list_path = os.path.join(list_root, 'multipie_' + str(testlist[index]) + '_train_list.txt')
            # # probe_list_path = os.path.join(list_root, 'multipie_gallery_test_list.txt')
            # gallery_list_path = os.path.join(list_root, 'multipie_gallery_train_list.txt')

            probe_embeddings, probe_ids = parse_list_file(probe_list_path, probe_root, feature_extractor) ## generated
            gallery_embeddings, gallery_ids = parse_list_file(gallery_list_path, gallery_root, feature_extractor) ## groundtruth 051-06 face
            if weight != 1:
                probe_embeddings_original, _ = parse_list_file(
                    probe_list_path, probe_root_original, feature_extractor, strip_NIR=False)
                probe_embeddings = [weight * item + (1 - weight) * probe_embeddings_original[idx] for idx, item in enumerate(probe_embeddings)]

            [rank_acc] = rank_accuracy(gallery_embeddings, gallery_ids, probe_embeddings, probe_ids, ranks=1, verbose=False)
            # is_same, similarity = full_comparison(gallery_embeddings, gallery_ids, probe_embeddings, probe_ids)
            # tpr1, tpr01, tpr001, _, _ = verification(is_same, similarity, verbose=False)

            rank_accs.append(rank_acc)
            # tpr1s.append(tpr1)
            # tpr01s.append(tpr01)
            # tpr001s.append(tpr001)

        printText = '{:.2f}% +- {:.2f}%'.format(np.mean(rank_accs), np.var(rank_accs))

        print('rank-1 accuracy: ' + printText)
        # print('TPR={:.2f}% +- {:.2f}%@FPR=1%'.format(np.mean(tpr1s), np.var(tpr1s)))
        # print('TPR={:.2f}% +- {:.2f}%@FPR=0.1%'.format(np.mean(tpr01s), np.var(tpr01s)))
        # print('TPR={:.2f}% +- {:.2f}%@FPR=0.01%'.format(np.mean(tpr001s), np.var(tpr001s)))

        resultList.append(printText)
    return resultList


def save_data(file_save_path, data_save):
    mode = 'a' if os.path.exists(file_save_path) else 'w'
    with open(file_save_path, mode) as f:
        f.write(data_save)


if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # os.chdir(os.path.dirname(os.path.abspath(__file__)))
    #
    # parser = argparse.ArgumentParser()
    # main(get_config(parser))


    # for i in range(11):
    #     # weight = 0.1 * i
    #     weight = 1
    #     print('=' * 50)
    #     print('weight = ' + str(weight))
    #     evaluate(probe_root='/home/jie.cao/main/model_output/NIR2VIS/casia',
    #              gallery_root='/home/jie.cao/main/dataset/CASIA-NIR-VIS-2.0/Jie/data',
    #              list_root='/home/jie.cao/main/dataset/CASIA-NIR-VIS-2.0/Protocol/test', weight=weight)

    # weight = 0.1 * i
    weight = 1
    print('=' * 50)


    # testlist = [15, 30, 45]
    testlist = [15, 30, 45, 60, 75, 90]

    path_root = "/data1/mandi.luo/work/FaceRotation/cjcode-2-v1/mainBig/model_output/FE_frontalization/MP/"
    gallery_root = '/data1/mandi.luo/dataset/Multi-PIE/data/'
    list_root = '/data1/mandi.luo/dataset/Multi-PIE/FS/'

    #############eval once############
    # test_model_epoch = 10
    # file_path = path_root + "evalResult-" + str(test_model_epoch) + "_weight" + str(weight) + ".txt"
    # probe_root = path_root + 'output-eval2train-eval-' + str(test_model_epoch) + '/'
    # data = evaluate(probe_root=probe_root, gallery_root=gallery_root,
    #                 list_root=list_root, weight=weight, testlist=testlist)
    #
    # for k in range(len(data)):
    #     data_save = "test_model_epoch " + str(test_model_epoch) + " rank-1 of angle " + str((k + 1) * 15) + " is " \
    #                 + str(data[k]) + "\n"
    #
    #     save_data(file_path, data_save)
    #############eval once############

    # #############eval iteration############
    for i in range(7, 8):
        print("test_model_epoch is " + str(i))
        data = evaluate(probe_root=path_root + 'output-eval' + str(i) + '/', gallery_root=gallery_root,
                        list_root=list_root, weight=weight, testlist=testlist)

        for k in range(len(data)):
            data_save = "test_model_epoch " + str(i) + " rank-1 of angle " + str((k + 1) * 15) + " is " \
                                + str(data[k]) + "\n"
            file_path = path_root + "evalResult-" + str(i) + "_weight" + str(weight) + ".txt"
            save_data(file_path, data_save)
    # #############eval iteration############



