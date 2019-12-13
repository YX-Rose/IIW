
import os
import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity


# verification
def full_comparison(gallery_embeddings, gallery_ids, probe_embeddings, probe_ids):

    gallery_embeddings = np.array(gallery_embeddings).squeeze()
    probe_embeddings = np.array(probe_embeddings).squeeze()

    similarity_matrix = cosine_similarity(gallery_embeddings, probe_embeddings)

    is_same = list()
    similarity = list()

    for gallery_idx in range(len(gallery_embeddings)):
        for probe_idx in range(len(probe_embeddings)):

            if gallery_ids[gallery_idx] == probe_ids[probe_idx]:
                is_same.append(1)
            else:
                is_same.append(0)
            similarity.append(similarity_matrix[gallery_idx, probe_idx])

    # spatial.distance.cosine
    return is_same, similarity


def verification(is_same, similarity, verbose=True):

    similarity = np.array(similarity)
    is_same = np.array(is_same)

    # ROC && AUC
    fpr, tpr, thresholds = roc_curve(is_same, similarity)
    auc = roc_auc_score(is_same, similarity)

    # # max ACC
    # maxacc = 0.0
    # for threshold in thresholds:
    #     total = 0
    #     for idx, score in enumerate(similarity):
    #         pred = 1 if score >= threshold else 0
    #         if pred == is_same[idx]:
    #             total += 1
    #     acc = float(total) / len(is_same)
    #     maxacc = maxacc if maxacc >= acc else acc
    #
    # # EER, caution when the numbers of the real pairs and fake pairs are not equal
    # positive = similarity[is_same == 1]
    # negtive = similarity[is_same == 0]
    # tmp_p = len(positive)
    # tmp_n = 0
    #
    # for threshold in thresholds:
    #     p = np.sum(positive <= threshold)
    #     n = np.sum(negtive >= threshold)
    #     if abs(p - n) < abs(tmp_p - tmp_n):
    #         tmp_p = p
    #         tmp_n = n
    #
    tpr1 = true_positive_rate(fpr, tpr, target_fpr=0.01)
    tpr01 = true_positive_rate(fpr, tpr, target_fpr=0.001)
    tpr001 = true_positive_rate(fpr, tpr, target_fpr=0.0001)
    eer = get_eer(tpr, fpr)

    if verbose:
        # print('accuracy: {:.2f}%'.format(100 * maxacc))
        print('TPR={:2f}%@FPR=1%'.format(100 * tpr1))
        print('TPR={:2f}%@FPR=0.1%'.format(100 * tpr01))
        print('TPR={:2f}%@FPR=0.01%'.format(100 * tpr001))
        print('AUC: {:.2f}%'.format(100 * auc))
        print('EER: {:.2f}%'.format(100 * eer))
        return tpr1, tpr01, tpr001, auc, eer

    else:
        output = list()
        # output.append(100 * maxacc)
        output.append(100 * tpr1)
        output.append(100 * tpr01)
        output.append(100 * tpr001)
        output.append(100 * eer)
        output.append(100 * auc)
        return output


def save_roc_curve(is_same, similarity, title='', save_image_name=''):

    fpr, tpr, thresholds = roc_curve(is_same, similarity)

    plt.title(title)
    plt.plot(fpr, tpr, 'b')

    plt.xscale('log')
    plt.xlim([0.01, 1])
    plt.ylim([0.75, 1])

    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    # plt.show()
    plt.savefig(os.path.join('/home/jie.cao/main/Downloads', save_image_name))
    return


def true_positive_rate(fpr, tpr, target_fpr):
    # to ensure small first
    fpr = np.sort(fpr)
    tpr = np.sort(tpr)

    residual = fpr - target_fpr
    residual[residual < 0] = np.inf
    floor_index = np.argmin(residual)
    floor_min = fpr[floor_index]

    residual = target_fpr - fpr
    residual[residual < 0] = np.inf
    # np.argmin will only return the first minimum
    ceil_min = np.min(residual)
    ceil_index = [index for index, element in enumerate(residual) if ceil_min == element][-1]

    factor = (target_fpr - ceil_min) / (floor_min - ceil_min)
    target_tpr = factor * tpr[floor_index] + (1 - factor) * tpr[ceil_index]
    return target_tpr


def get_eer(tpr, fpr):
    for i, fpr_point in enumerate(fpr):
        if tpr[i] >= (1 - fpr_point):
            idx = i
            break
    if tpr[idx] == tpr[idx+1]:
        return 1 - tpr[idx]
    else:
        return fpr[idx]


# recognition
def rank_accuracy(gallery_embeddings, gallery_ids, probe_embeddings, probe_ids, ranks=1, verbose=True):
    # [embedding1, embedding2, embedding3 ...] are extracted by Light CNN or other face recognizer
    # [ids1, ids2, ...] should has the same length with embeddings
    # output is [rank1, rank2, ...]

    gallery_embeddings = np.array(gallery_embeddings).squeeze()
    probe_embeddings = np.array(probe_embeddings).squeeze()

    similarity_matrix = cosine_similarity(gallery_embeddings, probe_embeddings)

    rank_acc = np.zeros(ranks)
    for i in range(len(probe_embeddings)):
        distance_list = 1 - similarity_matrix[:, i]
        for rank in range(1, ranks+1):
            min_indices = np.argsort(distance_list)[:rank]
            candidates = [gallery_ids[item] for item in min_indices]
            if probe_ids[i] in candidates:
                rank_acc[rank-1] += 1

    rank_acc = 100 * rank_acc / len(probe_embeddings)
    if verbose:
        for i in range(ranks):
            print('rank-' + str(i+1) + ' accuracy: {:.2f}%'.format(rank_acc[i]))
    else:
        return rank_acc

# recognition
def rank_accuracy_score_fuse(gallery_embeddings_origin, gallery_embeddings, gallery_ids, probe_embeddings_origin,
                             probe_embeddings, probe_ids, ranks=1, verbose=True, weight=0):
    # [embedding1, embedding2, embedding3 ...] are extracted by Light CNN or other face recognizer
    # [ids1, ids2, ...] should has the same length with embeddings
    # output is [rank1, rank2, ...]

    gallery_embeddings = np.array(gallery_embeddings).squeeze()
    probe_embeddings = np.array(probe_embeddings).squeeze()

    gallery_embeddings_origin = np.array(gallery_embeddings_origin).squeeze()
    probe_embeddings_origin = np.array(probe_embeddings_origin).squeeze()

    similarity_matrix_generated = cosine_similarity(gallery_embeddings, probe_embeddings)
    similarity_matrix_origin = cosine_similarity(gallery_embeddings_origin, probe_embeddings_origin)

    similarity_matrix = similarity_matrix_generated*weight + (1-weight)*similarity_matrix_origin

    rank_acc = np.zeros(ranks)
    for i in range(len(probe_embeddings)):
        distance_list = 1 - similarity_matrix[:, i]
        for rank in range(1, ranks+1):
            min_indices = np.argsort(distance_list)[:rank]
            candidates = [gallery_ids[item] for item in min_indices]
            if probe_ids[i] in candidates:
                rank_acc[rank-1] += 1

    rank_acc = 100 * rank_acc / len(probe_embeddings)
    if verbose:
        for i in range(ranks):
            print('rank-' + str(i+1) + ' accuracy: {:.2f}%'.format(rank_acc[i]))
    else:
        return rank_acc
