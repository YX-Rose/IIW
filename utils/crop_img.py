import cv2
import numpy as np
import math
import os
# from src import detect_faces
# import face_alignment
# from src import visualization_utils
# from PIL import Image


def read_list(list_path):
    img_list = []
    f5pt_list = []
    with open(list_path, 'r') as f:
        for line in f.readlines()[0:]:
            img_path = line.strip().split(' ')
            img_list.append(img_path[0])
    print('There are {} images..'.format(len(img_list)))
    return img_list


def read_ffp(ffp_path):
    ffp = np.loadtxt(ffp_path, dtype=float)
    ffp = np.reshape(ffp, (10,))
    return ffp


def transform(point, ang, src, dst):
    ang = -ang / 180 * np.pi
    x0 = point[0] - src[1] / 2
    y0 = point[1] - src[0] / 2
    xx = x0 * np.cos(ang) - y0 * np.sin(ang) + dst[1] / 2
    yy = x0 * np.sin(ang) + y0 * np.cos(ang) + dst[0] / 2
    return [xx, yy]


def bound_size(height, width, rotate_mat):
    abs_cos = abs(rotate_mat[0, 0])
    abs_sin = abs(rotate_mat[0, 1])
    bound_w = int(height * abs_sin + width * abs_cos) + 2
    bound_h = int(height * abs_cos + width * abs_sin) + 2
    return bound_h, bound_w


def guard(box, shape):
    box[0] = max(box[0], 0)
    box[1] = max(box[1], 0)
    box[2] = min(box[2], shape[1])
    box[3] = min(box[3], shape[0])
    return box


def align(img, ffp, img_size=128, padding=0):
    """
        align face
    Parameters:
    ----------
        img: numpy array, bgr order of shape (1, 3, h, w)
            input image
        ffp: numpy array, 10 x 1 (x1, y1, ... , x5, y5)
        img_size: default 128
        padding: default 0
    Retures:
    -------
        align_img:
            aligned face
    """
    # 68 landmark point
    # lefteye_center = [(ffp[36][0] + ffp[39][0]) / 2, (ffp[36][1] + ffp[39][1]) / 2]
    # righteye_center = [(ffp[42][0] + ffp[45][0]) / 2, (ffp[42][1] + ffp[45][1]) / 2]

    # five landmark point
    ang_tan = (float)(ffp[1] - ffp[3]) / (float)(ffp[0] - ffp[2] + 1e-6)  # (eyeLy - eyeRy) / (eyeLx - eyeRx)

    # 68 landmark point
    # ang_tan = (float)((lefteye_center[1] - righteye_center[1]) / (lefteye_center[0] - righteye_center[0]))

    ang = np.arctan(ang_tan) / np.pi * 180
    h = img.shape[0]
    w = img.shape[1]

    # five landmark point
    src_eye_center = [(ffp[0] + ffp[2]) / 2, (ffp[1] + ffp[3]) / 2]
    src_mouth_center = [(ffp[6] + ffp[8]) / 2, (ffp[7] + ffp[9]) / 2]

    # # 68 landmark point
    # src_eye_center = [(lefteye_center[0] + righteye_center[0]) / 2, (lefteye_center[1] + righteye_center[1]) / 2]
    # src_mouth_center = [(ffp[48][0] + ffp[54][0]) / 2, (ffp[48][1] + ffp[54][1]) / 2]

    rotate_mat = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    bound_h, bound_w = bound_size(h, w, rotate_mat)

    rotate_mat[0, 2] += (bound_w - w) / 2
    rotate_mat[1, 2] += (bound_h - h) / 2

    img_rotate = cv2.warpAffine(img, rotate_mat, (bound_w, bound_h), flags=cv2.INTER_CUBIC)
    dst_eye_center = transform(src_eye_center, ang, img.shape, img_rotate.shape)
    dst_mouth_center = transform(src_mouth_center, ang, img.shape, img_rotate.shape)

    ec_mc_y = 48.0  # 48.0
    ec_y = 40.0  # 40.0

    rate = 1.25

    dist_ec_mc = np.abs(dst_mouth_center[1] - dst_eye_center[1])
    scale = ec_mc_y / dist_ec_mc
    dist_y = (ec_y + padding) / scale
    crop_size = int(dist_y * 2 + dist_ec_mc)
    crop_x = int(dst_eye_center[0] - crop_size / 2)
    crop_y = int(dst_eye_center[1] - (dist_y * rate))
    crop_x_end = int(crop_x + crop_size - 1)
    crop_y_end = int(crop_y + crop_size - 1)

    box = [crop_x, crop_y, crop_x_end, crop_y_end]
    box = guard(box, img_rotate.shape)

    img_crop = np.zeros([crop_size, crop_size, img_rotate.shape[2]], dtype=np.uint8);
    img_crop[box[1] - crop_y:box[3] - crop_y, box[0] - crop_x:box[2] - crop_x, :] \
        = img_rotate[box[1]:box[3], box[0]:box[2], :]

    img_size = img_size + 2 * padding
    img_crop = cv2.resize(img_crop, (img_size, img_size))
    return img_crop


def list2colmatrix(pts_list):
    """
        convert list to column matrix
    Parameters:
    ----------
        pts_list:
            input list
    Retures:
    -------
        colMat:
    """
    assert len(pts_list) > 0
    colMat = []
    for i in range(len(pts_list)):
        colMat.append(pts_list[i][0])
        colMat.append(pts_list[i][1])
    colMat = np.matrix(colMat).transpose()
    return colMat


def find_tfrom_between_shapes(from_shape, to_shape):
    """
        find transform between shapes
    Parameters:
    ----------
        from_shape:
        to_shape:
    Retures:
    -------
        tran_m:
        tran_b:
    """
    assert from_shape.shape[0] == to_shape.shape[0] and from_shape.shape[0] % 2 == 0

    sigma_from = 0.0
    sigma_to = 0.0
    cov = np.matrix([[0.0, 0.0], [0.0, 0.0]])
    # compute the mean and cov
    from_shape_points = from_shape.reshape(from_shape.shape[0] / 2, 2)
    to_shape_points = to_shape.reshape(to_shape.shape[0] / 2, 2)
    mean_from = from_shape_points.mean(axis=0)
    mean_to = to_shape_points.mean(axis=0)

    for i in range(from_shape_points.shape[0]):
        temp_dis = np.linalg.norm(from_shape_points[i] - mean_from)
        sigma_from += temp_dis * temp_dis
        temp_dis = np.linalg.norm(to_shape_points[i] - mean_to)
        sigma_to += temp_dis * temp_dis
        cov += (to_shape_points[i].transpose() - mean_to.transpose()) * (from_shape_points[i] - mean_from)

    sigma_from = sigma_from / to_shape_points.shape[0]
    sigma_to = sigma_to / to_shape_points.shape[0]
    cov = cov / to_shape_points.shape[0]

    # compute the affine matrix
    s = np.matrix([[1.0, 0.0], [0.0, 1.0]])
    u, d, vt = np.linalg.svd(cov)

    if np.linalg.det(cov) < 0:
        if d[1] < d[0]:
            s[1, 1] = -1
        else:
            s[0, 0] = -1
    r = u * s * vt
    c = 1.0
    if sigma_from != 0:
        c = 1.0 / sigma_from * np.trace(np.diag(d) * s)

    tran_b = mean_to.transpose() - c * r * mean_from.transpose()
    tran_m = c * r

    return tran_m, tran_b


def extract_image_chips(img, points, desired_size=256, padding=0):
    """
        crop and align face
    Parameters:
    ----------
        img: numpy array, bgr order of shape (1, 3, n, m)
            input image
        points: numpy array, n x 10 (x1, x2 ... x5, y1, y2 ..y5)
        desired_size: default 256
        padding: default 0
    Retures:
    -------
        crop_imgs: list, n
            cropped and aligned faces
    """
    shape = points

    if padding > 0:
        padding = padding
    else:
        padding = 0
    # average positions of face points
    mean_face_shape_x = [0.3405, 0.6751, 0.5009, 0.3718, 0.6452]
    mean_face_shape_y = [0.3203, 0.3203, 0.5059, 0.6942, 0.6962]

    from_points = []
    to_points = []

    for i in range(len(shape) / 2):
        x = mean_face_shape_x[i] * desired_size + padding
        y = mean_face_shape_y[i] * desired_size + padding
        to_points.append([x, y])
        from_points.append([shape[2 * i], shape[2 * i + 1]])
    desired_size = desired_size + 2 * padding

    # convert the points to Mat
    from_mat = list2colmatrix(from_points)
    to_mat = list2colmatrix(to_points)

    # compute the similar transfrom
    tran_m, tran_b = find_tfrom_between_shapes(from_mat, to_mat)

    probe_vec = np.matrix([1.0, 0.0]).transpose()
    probe_vec = tran_m * probe_vec

    scale = np.linalg.norm(probe_vec)
    angle = 180.0 / math.pi * math.atan2(probe_vec[1, 0], probe_vec[0, 0])

    from_center = [(shape[0] + shape[2]) / 2.0, (shape[1] + shape[3]) / 2.0]
    to_center = [0, 0]
    to_center[1] = 0.3203 * (desired_size - 2 * padding) + padding
    to_center[0] = desired_size * 0.5

    ex = to_center[0] - from_center[0]
    ey = to_center[1] - from_center[1]

    rot_mat = cv2.getRotationMatrix2D((from_center[0], from_center[1]), -1 * angle, scale)
    rot_mat[0][2] += ex
    rot_mat[1][2] += ey

    crop_img = cv2.warpAffine(img, rot_mat, (desired_size, desired_size))

    return crop_img


if __name__ == '__main__':

    # config
    input_root = '/data1/xin.ma/datasets/VGGFace2/'
    pts_root = os.path.join(input_root, 'bb_landmark', 'loose_landmark_test.csv') #  face landmark
    output_root = os.path.join(input_root, 'test')
    ffps = {}
    with open(pts_root, 'r') as f:
        for line in f.readlines():
            image = line.strip().split(',')[0].replace('"','')
            ffp = line.strip().split(',')[1:]
            ffps[image] = ffp

    image_list = []
    with open(os.path.join(input_root, 'test.txt'), 'r') as f:
        for line in f.readlines():
            image_list.append(line.strip())

    error_imgs = []
    for image_path in image_list:
        print(image_path)
        image_name = image_path.split('.')[0]
        img = cv2.imread(os.path.join(input_root, 'test', image_path))
        # ffp_path = os.path.join(pts_root, image_path.split('.')[0] + '.pts')
        # ffp = read_ffp(ffp_path)
        ffp = np.array(ffps[image_name], np.float32).reshape(10,)
        try:
            img_align = align(img, ffp, 128, 0)
        except cv2.error:
            print('Occur at {}'.format(image_path))
            error_imgs.append(image_path)
            continue

        image_folder = image_name.split('/')[0]
        if not os.path.exists(os.path.join(output_root, image_folder)):
            os.makedirs(os.path.join(output_root, image_folder))
        cv2.imwrite(os.path.join(output_root, image_path), img_align)

    with open("/data1/xin.ma/datasets/VGGFace2/error_imgs.txt", 'w') as f:
        for i in error_imgs:
            f.writelines(i)
            f.writelines('\n')
    # get image path list

    # fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cpu')

    # root = "raw_data_bu/VIS/P0561"
    # img_list = sorted(util.make_dataset_list(root))
    # for img_path in img_list:
    #     input = io.imread(img_path)
    #     preds = fa.get_landmarks(input)
    #     # # preds = preds[0]
    #     # print(len(preds))
    #
    #     # img = Image.open(img_path)
    #     # img_crop = visualization_utils.show_bboxes(img, [], preds)
    #     # idx = idx + 1
    #     # save_path = 'crop/face_{:s}.jpeg'.format(str(idx))
    #     # img_crop.save(save_path)
    #
    #     # if preds != None:
    #     img = cv2.imread(img_path)
    #     for pred in preds:
    #         if abs(pred[8][1] - pred[27][1]) < 64:
    #             continue
    #         img_align = align(img, pred, img_size=236, padding=10)
    #
    #         save_path = img_path.replace('raw_data', 'dataset')
    #         dir = save_path[:-5]
    #         isExists = os.path.exists(dir)
    #         if not isExists:
    #             os.makedirs(dir)
    #             print(save_path + ' make successfully!')
    #         cv2.imwrite(save_path, img_align)

