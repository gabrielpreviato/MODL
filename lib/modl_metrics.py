import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K


def overlap(x1, w1, x2, w2):
    l1 = x1 - w1 / 2
    l2 = x2 - w2 / 2
    left = tf.where(K.greater(l1, l2), l1, l2)
    r1 = x1 + w1 / 2
    r2 = x2 + w2 / 2
    right = tf.where(K.greater(r1, r2), r2, r1)
    result = right - left
    return result


def iou(x_true, y_true, w_true, h_true, x_pred, y_pred, w_pred, h_pred, t, pred_confid_tf):
    x_true = K.expand_dims(x_true, 2)
    y_true = K.expand_dims(y_true, 2)
    w_true = K.expand_dims(w_true, 2)
    h_true = K.expand_dims(h_true, 2)
    x_pred = K.expand_dims(x_pred, 2)
    y_pred = K.expand_dims(y_pred, 2)
    w_pred = K.expand_dims(w_pred, 2)
    h_pred = K.expand_dims(h_pred, 2)

    # a = list(range(0, 8)) * 5
    a = list(range(0, 16)) * 10
    b = sorted([i for i in 16 * [j for j in range(0, 10)]])

    xoffset = K.expand_dims(tf.convert_to_tensor(np.asarray(
        a, dtype=np.float32)), 1)
    yoffset = K.expand_dims(tf.convert_to_tensor(np.asarray(
        b, dtype=np.float32)), 1)

    # xoffset = K.cast_to_floatx((np.tile(np.arange(side),side)))
    # yoffset = K.cast_to_floatx((np.repeat(np.arange(side),side)))
    x = tf.where(t, x_pred, K.zeros_like(x_pred))
    y = tf.where(t, y_pred, K.zeros_like(y_pred))
    w = tf.where(t, w_pred, K.zeros_like(w_pred))
    h = tf.where(t, h_pred, K.zeros_like(h_pred))

    ow = overlap(x + xoffset, w * 4*256., x_true + xoffset, w_true * 4*256.)
    oh = overlap(y + yoffset, h * 4*160., y_true + yoffset, h_true * 4*160.)

    ow = tf.where(K.greater(ow, 0), ow, K.zeros_like(ow))
    oh = tf.where(K.greater(oh, 0), oh, K.zeros_like(oh))
    intersection = ow * oh
    union = w * 4*256. * h *4* 160. + w_true * 4*256. * h_true *4* 160. - intersection + K.epsilon()  # prevent div 0

    #
    # find best iou among bboxs
    # iouall shape=(-1, bnum*gridcells)
    iouall = intersection / union
    obj_count = K.sum(tf.where(t, K.ones_like(x_true), K.zeros_like(x_true)))

    ave_iou = K.sum(iouall) / (obj_count + 0.0000001)
    recall_t = K.greater(iouall, 0.5)
    # recall_count = K.sum(tf.select(recall_t, K.ones_like(iouall), K.zeros_like(iouall)))

    fid_t = K.greater(pred_confid_tf, 0.3)
    recall_count_all = K.sum(tf.where(fid_t, K.ones_like(iouall), K.zeros_like(iouall)))

    #  
    obj_fid_t = tf.logical_and(fid_t, t)
    obj_fid_t = tf.logical_and(fid_t, recall_t)
    effevtive_iou_count = K.sum(tf.where(obj_fid_t, K.ones_like(iouall), K.zeros_like(iouall)))

    recall = effevtive_iou_count / (obj_count + 0.00000001)
    precision = effevtive_iou_count / (recall_count_all + 0.0000001)
    return ave_iou, recall, precision, obj_count, intersection, union, ow, oh, x, y, w, h

def recall(y_true, y_pred):
    truth_conf_tensor = K.expand_dims(y_true[:, :, 0], 2)  # tf.slice(y_true, [0, 0, 0], [-1,-1, 0])
    truth_xy_tensor = y_true[:, :, 1:3]  # tf.slice(y_true, [0, 0, 1], [-1,-1, 2])
    truth_wh_tensor = y_true[:, :, 3:5]  # tf.slice(y_true, [0, 0, 3], [-1, -1, 4])

    pred_conf_tensor = K.expand_dims(y_pred[:, :, 0], 2)  # tf.slice(y_pred, [0, 0, 0], [-1, -1, 0])
    # pred_conf_tensor = K.tanh(pred_conf_tensor)
    pred_xy_tensor = y_pred[:, :, 1:3]  # tf.slice(y_pred, [0, 0, 1], [-1, -1, 2])
    pred_wh_tensor = y_pred[:, :, 3:5]  # tf.slice(y_pred, [0, 0, 3], [-1, -1, 4])

    tens = K.greater(truth_conf_tensor, 0.5)

    ave_iou, recall, precision, obj_count, intersection, union, ow, oh, x, y, w, h = iou(truth_xy_tensor[:, :, 0],
                                                                                         truth_xy_tensor[:, :, 1],
                                                                                         truth_wh_tensor[:, :, 0],
                                                                                         truth_wh_tensor[:, :, 1],
                                                                                         pred_xy_tensor[:, :, 0],
                                                                                         pred_xy_tensor[:, :, 1],
                                                                                         pred_wh_tensor[:, :, 0],
                                                                                         pred_wh_tensor[:, :, 1],
                                                                                         tens, pred_conf_tensor)
    return recall


def precision(y_true, y_pred):
    truth_conf_tensor = K.expand_dims(y_true[:, :, 0], 2)  # tf.slice(y_true, [0, 0, 0], [-1,-1, 0])
    truth_xy_tensor = y_true[:, :, 1:3]  # tf.slice(y_true, [0, 0, 1], [-1,-1, 2])
    truth_wh_tensor = y_true[:, :, 3:5]  # tf.slice(y_true, [0, 0, 3], [-1, -1, 4])

    pred_conf_tensor = K.expand_dims(y_pred[:, :, 0], 2)  # tf.slice(y_pred, [0, 0, 0], [-1, -1, 0])
    # pred_conf_tensor = K.tanh(pred_conf_tensor)
    pred_xy_tensor = y_pred[:, :, 1:3]  # tf.slice(y_pred, [0, 0, 1], [-1, -1, 2])
    pred_wh_tensor = y_pred[:, :, 3:5]  # tf.slice(y_pred, [0, 0, 3], [-1, -1, 4])

    tens = K.greater(truth_conf_tensor, 0.5)

    ave_iou, recall, precision, obj_count, intersection, union, ow, oh, x, y, w, h = iou(truth_xy_tensor[:, :, 0],
                                                                                         truth_xy_tensor[:, :, 1],
                                                                                         truth_wh_tensor[:, :, 0],
                                                                                         truth_wh_tensor[:, :, 1],
                                                                                         pred_xy_tensor[:, :, 0],
                                                                                         pred_xy_tensor[:, :, 1],
                                                                                         pred_wh_tensor[:, :, 0],
                                                                                         pred_wh_tensor[:, :, 1],
                                                                                         tens, pred_conf_tensor)
    return precision


def iou_metric(y_true, y_pred):
    truth_conf_tensor = K.expand_dims(y_true[:, :, 0], 2)  # tf.slice(y_true, [0, 0, 0], [-1,-1, 0])
    truth_xy_tensor = y_true[:, :, 1:3]  # tf.slice(y_true, [0, 0, 1], [-1,-1, 2])
    truth_wh_tensor = y_true[:, :, 3:5]  # tf.slice(y_true, [0, 0, 3], [-1, -1, 4])

    pred_conf_tensor = K.expand_dims(y_pred[:, :, 0], 2)  # tf.slice(y_pred, [0, 0, 0], [-1, -1, 0])
    # pred_conf_tensor = K.tanh(pred_conf_tensor)
    pred_xy_tensor = y_pred[:, :, 1:3]  # tf.slice(y_pred, [0, 0, 1], [-1, -1, 2])
    pred_wh_tensor = y_pred[:, :, 3:5]  # tf.slice(y_pred, [0, 0, 3], [-1, -1, 4])

    tens = K.greater(truth_conf_tensor, 0.5)

    ave_iou, recall, precision, obj_count, intersection, union, ow, oh, x, y, w, h = iou(truth_xy_tensor[:, :, 0],
                                                                                         truth_xy_tensor[:, :, 1],
                                                                                         truth_wh_tensor[:, :, 0],
                                                                                         truth_wh_tensor[:, :, 1],
                                                                                         pred_xy_tensor[:, :, 0],
                                                                                         pred_xy_tensor[:, :, 1],
                                                                                         pred_wh_tensor[:, :, 0],
                                                                                         pred_wh_tensor[:, :, 1],
                                                                                         tens, pred_conf_tensor)
    return ave_iou
