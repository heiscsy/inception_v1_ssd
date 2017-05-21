from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.slim.python.slim.nets import inception_v1
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.python.ops import variable_scope
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.contrib.layers import xavier_initializer_conv2d
from tensorflow.contrib.slim.python.slim import learning

import tensorflow as tf
import numpy as np

lr_decay = 0.1
learning_rate_init = 0.1
learning_rate_decrease_step = 10000
class_num = 21

batch_size = 32
kEpsilon = 2.220446049250313e-16


def conv_layer(input_layer, shape, stride, name, scope, is_training, reuse, padding='SAME'):
    with variable_scope.variable_scope(scope, reuse=reuse):
        w = tf.get_variable(name+'/weights', shape=shape, initializer=xavier_initializer_conv2d())
        net = tf.nn.conv2d(input=input_layer, filter=w, strides=stride, padding=padding)
        net = tf.contrib.layers.batch_norm(net, decay=0.99, is_training=is_training, center = True, scale=True, reuse=reuse,
            trainable=True, scope=name+'/BatchnNorm')
        net = tf.nn.relu(net)
    return net

def calculate_overlap(gt, db):
    return np.max(0, (np.max(gt[0], db[0])-np.min(gt[2], db[2]))*(np.max(gt[1], db[1])-np.min(gt[3], db[3])))

class SSD:
    image = None
    bbox = None
    label = None
    reuse = None
    is_training = None
    scope = None
    step = None
    train_op = None
    assert_op = None
    feature_map = []
    fk = [38, 19, 10, 5, 3, 1]
    db_size = [4, 6, 6, 6, 6, 6]
    ar = [[1,2,0.5], [1,2,0.5, 3, 1/3]]
    expand_size = [np.sqrt(0.2*0.34), np.sqrt(0.34*0.48), np.sqrt(0.48*0.62),
        np.sqrt(0.62*0.76), np.sqrt(0.76*0.9), np.sqrt(1.04*0.9)]
    db = [0.2, 0.34, 0.48, 0.62, 0.76, 0.9]
    default_box = []
    feature_map_label = []
    pos_mask = []

    def __init__(self, reuse=None, is_training = True, scope = ''):
        self.image = tf.placeholder(tf.float32, shape=[batch_size, 300, 300, 3])
        self.bbox = tf.placeholder(tf.float32, shape=[None, None, 4])
        self.label = tf.placeholder(tf.int64, shape=[None])

        self.feature_map_label.append(tf.placeholder(tf.float32, shape=[batch_size, 38, 38, 100]))
        self.feature_map_label.append(tf.placeholder(tf.float32, shape=[batch_size, 19, 19, 150]))
        self.feature_map_label.append(tf.placeholder(tf.float32, shape=[batch_size, 10, 10, 150]))
        self.feature_map_label.append(tf.placeholder(tf.float32, shape=[batch_size, 5, 5, 150]))
        self.feature_map_label.append(tf.placeholder(tf.float32, shape=[batch_size, 3, 3, 150]))
        self.feature_map_label.append(tf.placeholder(tf.float32, shape=[batch_size, 1, 1, 150]))

        self.pos_mask.append(tf.placeholder(tf.float32, shape=[batch_size, 38, 38, 4]))
        self.pos_mask.append(tf.placeholder(tf.float32, shape=[batch_size, 19, 19, 6]))
        self.pos_mask.append(tf.placeholder(tf.float32, shape=[batch_size, 10, 10, 6]))
        self.pos_mask.append(tf.placeholder(tf.float32, shape=[batch_size, 5, 5, 6]))
        self.pos_mask.append(tf.placeholder(tf.float32, shape=[batch_size, 3, 3, 6]))
        self.pos_mask.append(tf.placeholder(tf.float32, shape=[batch_size, 1, 1, 6]))

        self.pos_box_num = tf.placeholder(tf.int32, shape=[batch_size])
        self.step = tf.placeholder(tf.int64, shape=[])
        self.calculateDefaultBoxParam()

        self.reuse = reuse
        self.is_training = is_training
        self.scope = scope

        self.build_graph()
        self.loss = self.build_loss(self.scope)
        self.train_op = self.build_train_op(self.loss, self.step)

    def calculateDefaultBoxParam(self):
        for l in range(6):
            db_loc = np.zeros([self.fk[l], self.fk[l], self.db_size[l]*4])
            index_linespace = range(self.fk[l])
            xv, yv = np.meshgrid(index_linespace, index_linespace)
            yv = (yv+0.5) / self.fk[l]
            xv = (xv+0.5) / self.fk[l]
            db_loc[:,:,self.db_size[l]*0:self.db_size[l]*1] = np.stack(
                [yv]*self.db_size[l], axis = 2)
            db_loc[:,:,self.db_size[l]*1:self.db_size[l]*2] = np.stack(
                [xv]*self.db_size[l], axis = 2)
            w = [self.db[i]*np.sqrt(self.ar[1][i]) for i in range(self.db_size[l]-1)]
            w.append(self.expand_size[l])
            w = np.reshape(w, [1, 1, self.db_size[l]])
            h = [self.db[i]/np.sqrt(self.ar[1][i]) for i in range(self.db_size[l]-1)]
            h.append(self.expand_size[l])
            h = np.reshape(h, [1, 1, self.db_size[l]])
            db_loc[:,:,self.db_size[l]*2:self.db_size[l]*3] = np.tile(
                w, (self.fk[l], self.fk[l], 1))
            db_loc[:,:,self.db_size[l]*3:self.db_size[l]*4] = np.tile(
                h, (self.fk[l], self.fk[l], 1))
            self.default_box.append(db_loc)



    def build_loss(self, scope):
        with variable_scope.variable_scope(scope, reuse=self.reuse):
            for idx in range(6):
                if idx == 0:
                    db_size = 4
                else:
                    db_size = 6
                conf_pred, loc_pred = tf.split(self.feature_map[idx], [db_size*class_num, db_size*4], 3)
                conf_labl, loc_labl = tf.split(self.feature_map_label[idx], [db_size*class_num, db_size*4], 3)
                conf_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=conf_labl, logits=conf_pred))
                tf.add_to_collection(tf.GraphKeys.LOSSES, conf_loss)

                x_pred, y_pred, w_pred, h_pred = tf.split(loc_pred, [db_size, db_size, db_size, db_size], 3)
                x_labl, y_labl, w_labl, h_labl = tf.split(loc_labl, [db_size, db_size, db_size, db_size], 3)
                x_dbox, y_dbox, w_dbox, h_dbox = tf.split(self.default_box[idx],
                    [db_size, db_size, db_size, db_size], 2)
                x_dbox = tf.tile(tf.expand_dims(tf.cast(tf.convert_to_tensor(x_dbox),dtype=tf.float32), 0), 
                    [tf.shape(x_labl)[0], 1, 1, 1])
                y_dbox = tf.tile(tf.expand_dims(tf.cast(tf.convert_to_tensor(y_dbox),dtype=tf.float32), 0), 
                    [tf.shape(x_labl)[0], 1, 1, 1])
                w_dbox = tf.tile(tf.expand_dims(tf.cast(tf.convert_to_tensor(w_dbox),dtype=tf.float32), 0), 
                    [tf.shape(x_labl)[0], 1, 1, 1])
                h_dbox = tf.tile(tf.expand_dims(tf.cast(tf.convert_to_tensor(h_dbox),dtype=tf.float32), 0), 
                    [tf.shape(x_labl)[0], 1, 1, 1])

                w_pred = tf.where(tf.equal(w_pred, [0]), tf.ones_like(w_pred) * kEpsilon, w_pred); 
                h_pred = tf.where(tf.equal(h_pred, [0]), tf.ones_like(h_pred) * kEpsilon, h_pred);
                g_cx = tf.div((x_labl-x_dbox), w_dbox)
                g_cy = tf.div((y_labl-y_dbox), h_dbox)
                g_w = tf.log(tf.div(w_labl, w_dbox))
                g_h = tf.log(tf.div(h_labl, h_dbox))
                self.assert_op = tf.assert_greater(w_pred, tf.constant(0, dtype=tf.float32))
                l_cx = tf.div((x_pred-x_dbox), w_dbox)
                l_cy = tf.div((y_pred-y_dbox), h_dbox)
                l_w = tf.log(tf.div(w_pred, w_dbox))
                l_h = tf.log(tf.div(h_pred, h_dbox))
                # l_w = tf.where(tf.is_inf(l_w), tf.ones_like(l_w) * kEpsilon, l_w); 
                # l_h = tf.where(tf.is_inf(l_h), tf.ones_like(l_h) * kEpsilon, l_h);

                cx_loss = self.smoothL1(tf.multiply((l_cx-g_cx),self.pos_mask[idx]))
                #cx_loss = tf.reduce_mean(tf.multiply((l_cx-g_cx),self.pos_mask[idx]))
                tf.add_to_collection(tf.GraphKeys.LOSSES, cx_loss)
                cy_loss = self.smoothL1(tf.multiply((l_cy-g_cy),self.pos_mask[idx]))
                tf.add_to_collection(tf.GraphKeys.LOSSES, cy_loss)
                w_loss = self.smoothL1(tf.multiply((l_w-g_w),self.pos_mask[idx]))
                tf.add_to_collection(tf.GraphKeys.LOSSES, w_loss)
                h_loss = self.smoothL1(tf.multiply((l_h-g_h),self.pos_mask[idx]))
                tf.add_to_collection(tf.GraphKeys.LOSSES, h_loss)

            neg_loss = self.hardNegtiveMining(self.pos_box_num, self.feature_map_label, self.pos_mask)
            tf.add_to_collection(tf.GraphKeys.LOSSES, neg_loss)
            total_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.LOSSES))
        return total_loss

    def build_train_op(self, loss, train_step):
        learning_rate = lr_decay ** (train_step/learning_rate_decrease_step) * learning_rate_init
        momentum = 1 - learning_rate
        optimizer = tf.train.MomentumOptimizer(learning_rate = learning_rate, momentum=momentum)
        train_op = learning.create_train_op(loss, optimizer = optimizer)
        return train_op

    def build_graph(self):
        with arg_scope(inception_v1.inception_v1_arg_scope()):
            #with variable_scope.variable_scope(None, 'InceptionV1', [self.image, 21], reuse=None) as scope:
                with arg_scope([layers_lib.batch_norm, layers_lib.dropout], is_training=True):
                    logit, endpoints = inception_v1.inception_v1_base(self.image)
                    net = endpoints['Mixed_3c']
                    self.feature_map.append(conv_layer(
                        net, [3,3,480,4*(class_num+4)], [1,1,1,1], 'FeatureMap_1', 'SSD', self.is_training, self.reuse))
                    net = endpoints['Mixed_4f']
                    net = conv_layer(net, [3,3,832,1024], [1,1,1,1], 'conv6', 'SSD', self.is_training, self.reuse)
                    net = conv_layer(net, [1,1,1024,1024], [1,1,1,1], 'conv7', 'SSD', self.is_training, self.reuse)
                    self.feature_map.append(conv_layer(
                        net, [3,3,1024,6*(class_num+4)], [1,1,1,1], 'FeatureMap_2', 'SSD', self.is_training, self.reuse))
                    net = conv_layer(net, [1,1,1024,256], [1,1,1,1], 'conv8_1', 'SSD', self.is_training, self.reuse)
                    net = conv_layer(net, [3,3,256,512], [1,2,2,1], 'conv8_2', 'SSD', self.is_training, self.reuse)
                    self.feature_map.append(conv_layer(
                        net, [3,3,512,6*(class_num+4)], [1,1,1,1], 'FeatureMap_3', 'SSD', self.is_training, self.reuse))
                    net = conv_layer(net, [1,1,512,128], [1,1,1,1], 'conv9_1', 'SSD', self.is_training, self.reuse)
                    net = conv_layer(net, [3,3,128,256], [1,2,2,1], 'conv9_2', 'SSD', self.is_training, self.reuse)
                    self.feature_map.append(conv_layer(
                        net, [3,3,256,6*(class_num+4)], [1,1,1,1], 'FeatureMap_4', 'SSD', self.is_training, self.reuse))
                    net = conv_layer(net, [1,1,256,128], [1,1,1,1], 'conv10_1', 'SSD', self.is_training, self.reuse)
                    net = conv_layer(net, [3,3,128,256], [1,1,1,1], 'conv10_2', 'SSD', self.is_training, self.reuse,padding='VALID')
                    self.feature_map.append(conv_layer(
                        net, [3,3,256,6*(class_num+4)], [1,1,1,1], 'FeatureMap_5', 'SSD', self.is_training, self.reuse))
                    net = conv_layer(net, [1,1,256,128], [1,1,1,1], 'conv11_1', 'SSD', self.is_training, self.reuse)
                    net = conv_layer(net, [3,3,128,256], [1,1,1,1], 'conv11_2', 'SSD', self.is_training, self.reuse, padding='VALID')
                    self.feature_map.append(conv_layer(
                        net, [1,1,256,6*(class_num+4)], [1,1,1,1], 'FeatureMap_6', 'SSD', self.is_training, self.reuse))

    def detectOverlap(self, gt):
        result_box = []
        for l in range(6):
            f_k = self.fk[l]
            for i in range(f_k):
                for j in range(f_k):
                    cx = (i+0.5)/f_k
                    cy = (i+0.5)/f_k
                    a_r = None
                    if not(l==0):
                        a_r = self.ar[1]
                    else:
                        a_r = self.ar[0]
                    w = []
                    h = []
                    for db_idx in range(len(a_r)):
                        w.append(self.db[l] * np.sqrt(a_r[db_idx]))
                        h.append(self.db[l] / np.sqrt(a_r[db_idx]))
                    w.append(self.expand_size[l])
                    h.append(self.expand_size[l])

                    for db_idx in range(len(w)):
                        db = [cx-w[db_idx], cy-h[db_idx], cx+w[db_idx], cx+h[db_idx]]
                        if calculate_overlap(gt, db)>0.5:
                            result_box.append([l, i, j, db_idx])
        return result_box

    def smoothL1(self, diff):
        loss = tf.zeros(tf.shape(diff))
        mask_l2 = tf.cast(tf.less(tf.abs(diff), tf.constant(1.0, dtype =tf.float32)), tf.float32)
        mask_l1 = tf.constant(1.0, dtype = tf.float32)-mask_l2
        loss = tf.add(loss, tf.multiply(mask_l2, tf.square(diff)*0.5))
        loss = tf.add(loss, tf.multiply(mask_l1, tf.abs(diff)-0.5))
        return tf.reduce_mean(loss)






    def hardNegtiveMining(self, number, feature_map, pos_mask):
        batch_num = tf.shape(feature_map[0])[0]
        max_conf_list = []
        for batch_idx in range(batch_size):
            neg_loss = []
            neg_label_construct = [0.0]*21
            neg_label_construct[0] = 1.0
            neg_label = tf.constant(neg_label_construct)
            for idx in range(6):
                db_size = self.db_size[idx]
                fk = self.fk[idx]
                pred, _ = tf.split(self.feature_map[idx], [db_size*class_num, db_size*4], 3)
                # pred = tf.reshape(pred, [batch_size, fk*fk*db_size, class_num])
                pred = tf.reshape(pred[idx], [fk*fk*db_size, class_num])
                labl = tf.tile(tf.expand_dims(tf.expand_dims(neg_label, 0),0),
                    [fk*fk*db_size, 1]) 

                conf = tf.multiply(tf.nn.softmax_cross_entropy_with_logits(labels = labl, logits = pred),
                                1.0 - tf.reshape(pos_mask[idx][batch_idx], [fk*fk*db_size]))
                # conf = tf.multiply(tf.nn.softmax_cross_entropy_with_logits(labels = labl, logits = pred),
                #                 1.0 - tf.reshape(pos_mask[idx][batch_idx], [batch_size, fk*fk*db_size]))
                max_conf_list.append(conf)
            max_conf = tf.concat(max_conf_list, axis=1)
            value, _ = tf.nn.top_k(max_conf, k = number[batch_idx]*3, sorted=False)
        max_conf_list.append(value)
        return tf.reduce_mean(max_conf_list)

        """
        conf_ = []
        for idx in range(6):
            for i in range(self.fk[idx]):
                for j in range(self.fk[idx]):
                    db_size = 6
                    if idx == 0:
                        db_size = 4
                    for db in range(db_size):
                        conf_.append(np.max(feature_map[0, i, j, db*24+0:db*24+20]))
        result_conf_idx = np.array(conf_).argsort()[-number:]
        result_box = []
        box_num = [38*38*4, 19*19*6, 10*10*6, 5*5*6, 3*3*6, 1*1*6]
        box_idx_h = [38*38*4]*6
        box_idx_l = [0]*6
        for i in range(5):
            box_idx_h[i+1] = box_idx_h[i]+box_num[i+1]
            box_idx_l[i+1] = box_idx_h[i]

        for box_idx in result_conf_idx:
            for l in range(6):
                if box_idx<box_idx_h[l] and box_idx>box_idx_l[l]:
                    residue = box_idx-box_idx_l[l]
                    if l==0:
                        db_size=4
                    else:
                        db_size=6
                    db_idx = residue % db_size
                    residue = (residue-db_idx) // db_size
                    i = residue // self.fk[l]
                    j = residue % self.fk[l]
                    result_box.append([l,i,j,db_idx])
        return result_box
        """
