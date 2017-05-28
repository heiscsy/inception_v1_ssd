from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf

from tensorflow.contrib.slim.python.slim.data import dataset
from tensorflow.contrib.slim.python.slim.data import dataset_data_provider
from tensorflow.contrib.slim.python.slim.data import tfexample_decoder
from tensorflow.contrib.slim.python.slim.queues import QueueRunners

import ssd
import numpy as np
import struct

import utils
class_num = 21

kBatchSize = 32
kTotalSteps = 100000
kTrainFilePrefix = '/media/yuhanlong/data/VOCdevkit/VOC2012/TensorflowData2/train-'
checkpoint_file = './inception_v1.ckpt'



class SSDTrain:
    data_sources_train = []
    data_sources_val = []
    fk = [38, 19, 10, 5, 3, 1]
    db_size = [4, 6, 6, 6, 6, 6]
    ar = [[1,2,0.5], 
        [1,2,0.5, 3, 1/3], 
        [1,2,0.5, 3, 1/3], 
        [1,2,0.5, 3, 1/3], 
        [1,2,0.5, 3, 1/3], 
        [1,2,0.5, 3, 1/3]]
    expand_size = [np.sqrt(0.2*0.34), np.sqrt(0.34*0.48), np.sqrt(0.48*0.62),
        np.sqrt(0.62*0.76), np.sqrt(0.76*0.9), np.sqrt(1.04*0.9)]
    db = [0.2, 0.34, 0.48, 0.62, 0.76, 0.9]


    def __init__(self):
        data_sources_train = [kTrainFilePrefix+str(num) for num in range(16)]
        data_sources_val = [kTrainFilePrefix+str(num) for num in range(16, 18)]
        self.train_set = self.get_batch_preprocess(
            self.create_dataset(data_sources_train), kBatchSize)
        self.train_ssd = ssd.SSD(reuse = None, is_training=True, scope='Training')
        
        # self.feature_map_label_tf, self.feature_map_mask_tf, self.pos_box_num_tf = self.encodeBboxTensor(
        #     self.train_set[1], self.train_set[2], self.train_set[3], 
        #     self.train_set[4], self.train_set[5], self.train_set[6])
        # dataset = self.create_dataset()
    
    def create_dataset(self, data_source):
        keys_to_features = {
        'image/encoded': tf.FixedLenFeature(shape=(), dtype=tf.string, default_value=''),
        'image/format': tf.FixedLenFeature(shape=(), dtype=tf.string, default_value='png'),
        'image/object/bbox/xmin': tf.FixedLenFeature(shape=(), dtype= tf.string, default_value=''),
        'image/object/bbox/xmax': tf.FixedLenFeature(shape=(), dtype= tf.string, default_value=''),
        'image/object/bbox/ymin': tf.FixedLenFeature(shape=(), dtype= tf.string, default_value=''),
        'image/object/bbox/ymax': tf.FixedLenFeature(shape=(), dtype= tf.string, default_value=''),
        'image/object/bbox/label': tf.FixedLenFeature(shape=(), dtype = tf.string, default_value=''),
        'image/object/size': tf.FixedLenFeature(shape=(), dtype = tf.int64, default_value=0)
        }

        items_to_handlers = {
        'image': tfexample_decoder.Image(),
        'xmin': tfexample_decoder.Tensor('image/object/bbox/xmin'),
        'xmax': tfexample_decoder.Tensor('image/object/bbox/xmax'),
        'ymin': tfexample_decoder.Tensor('image/object/bbox/ymin'),
        'ymax': tfexample_decoder.Tensor('image/object/bbox/ymax'),
        #'bbox': tfexample_decoder.BoundingBox(prefix='image/object/bbox/'),
        'label': tfexample_decoder.Tensor('image/object/bbox/label'),
        'size': tfexample_decoder.Tensor('image/object/size')
        }

        decoder = tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)
        return dataset.Dataset(data_sources=data_source, reader = tf.TFRecordReader,
            decoder=decoder, num_samples = 1, items_to_descriptions=None)

    def process_image_and_label(self, image, label):
        image_mean = tf.constant([104.0,117.0,123.0])
        image = tf.subtract(
            tf.cast(tf.image.resize_images(image,[300,300]),tf.float32)
            , image_mean)
        #label = tf.cast(tf.image.resize_image_with_crop_or_pad(label,300,300), tf.int32)
        #label = tf.one_hot(label, 21, on_value=1.0, off_value=0.0)
        #label = array_ops.squeeze(label, [4])
        return image, label

    def get_batch_preprocess(self, dataset, batch_size):
        provider = dataset_data_provider.DatasetDataProvider(dataset)
        image, label, xmin, xmax, ymin, ymax, size = provider.get(
            ['image','label', 'xmin','xmax','ymin','ymax','size'])
        #label = array_ops.squeeze(label, [2])
        image, label = self.process_image_and_label(image, label)
        image_b, label_b, xmin_b, xmax_b, ymin_b, ymax_b, size_b = tf.train.batch(
            [image, label, xmin, xmax, ymin, ymax, size],
            batch_size = batch_size,
            num_threads = 4,
            capacity = 1024)
        return image_b, label_b, xmin_b, xmax_b, ymin_b, ymax_b, size_b
    """
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
    """
    def calculateDefaultBoxParam(self):
        db_dim_list = []
        for l in range(6):
            db_loc = np.zeros([self.fk[l], self.fk[l], self.db_size[l]*4])
            index_linespace = range(self.fk[l])
            xv, yv = np.meshgrid(index_linespace, index_linespace)
            yv = (yv+0.5) / self.fk[l]
            xv = (xv+0.5) / self.fk[l]
            db_loc[:,:,self.db_size[l]*0:self.db_size[l]*1] = np.stack(
                [xv]*self.db_size[l], axis = 2)
            db_loc[:,:,self.db_size[l]*1:self.db_size[l]*2] = np.stack(
                [yv]*self.db_size[l], axis = 2)
            w = [self.db[l]*np.sqrt(self.ar[l][i]) for i in range(self.db_size[l]-1)]
            w.append(self.expand_size[l])
            h = [self.db[l]/np.sqrt(self.ar[l][i]) for i in range(self.db_size[l]-1)]
            h.append(self.expand_size[l])
            db_loc[:,:,self.db_size[l]*2:self.db_size[l]*3] = w
            db_loc[:,:,self.db_size[l]*3:self.db_size[l]*4] = h
            db_dim = np.zeros([self.fk[l], self.fk[l], self.db_size[l]*4]).astype(np.float32)
            db_dim[:,:,self.db_size[l]*0:self.db_size[l]*1] = db_loc[
                :,:,self.db_size[l]*0:self.db_size[l]*1] - 0.5 * db_loc[
                :,:,self.db_size[l]*2:self.db_size[l]*3]

            db_dim[:,:,self.db_size[l]*1:self.db_size[l]*2] = db_loc[
                :,:,self.db_size[l]*1:self.db_size[l]*2] - 0.5 * db_loc[
                :,:,self.db_size[l]*3:self.db_size[l]*4]
 
            db_dim[:,:,self.db_size[l]*2:self.db_size[l]*3] = db_loc[
                :,:,self.db_size[l]*0:self.db_size[l]*1] + 0.5 * db_loc[
                :,:,self.db_size[l]*2:self.db_size[l]*3]

            db_dim[:,:,self.db_size[l]*3:self.db_size[l]*4] = db_loc[
                :,:,self.db_size[l]*1:self.db_size[l]*2] + 0.5 * db_loc[
                :,:,self.db_size[l]*3:self.db_size[l]*4]
            db_dim_list.append(db_dim)
        return db_dim_list

    def encodeBboxTensor(self, xmin, ymin, xmax, ymax, label, size):
        # label_shape = tf.concat([tf.expand_dims([-1]*kBatchSize, 0), 
        #     tf.expand_dims(tf.cast(size, dtype=tf.int32), 0)], 0)
        # label_shape = tf.transpose(label_shape)

        feature_map_label = [[],[],[],[],[],[]]
        feature_map_mask = [[],[],[],[],[],[]]
        pos_num_batch = []
        for batch_idx in range(kBatchSize): 
            fmt = "<%df" % (len(xmin[batch_idx]) /4)
            xmin_list = list(struct.unpack(fmt, xmin[batch_idx]))
            xmax_list = list(struct.unpack(fmt, xmax[batch_idx]))
            ymin_list = list(struct.unpack(fmt, ymin[batch_idx]))
            ymax_list = list(struct.unpack(fmt, ymax[batch_idx]))
            fmt = "<%di" % (len(xmin[batch_idx]) /4)
            label_list = list(struct.unpack(fmt, label[batch_idx]))
            # label_list = tf.reshape(tf.decode_raw(label, tf.float32), tf.cast(
            #     tf.expand_dims(size[batch_idx],0), dtype=tf.int32))
            # label_list = tf.cast(tf.reshape(label, size), tf.float32)
            pos_num = []
            default_box_dim = self.calculateDefaultBoxParam()
            for idx in range(6):
                gt_xmin = np.reshape(np.stack([xmin_list]*self.db_size[idx], axis = 1), [-1])
                gt_xmax = np.reshape(np.stack([xmax_list]*self.db_size[idx], axis = 1), [-1])
                gt_ymin = np.reshape(np.stack([ymin_list]*self.db_size[idx], axis = 1), [-1])
                gt_ymax = np.reshape(np.stack([ymax_list]*self.db_size[idx], axis = 1), [-1])
                gt_xmin = np.ones([self.fk[idx], self.fk[idx], np.shape(gt_xmin)[-1]])*gt_xmin
                gt_xmax = np.ones([self.fk[idx], self.fk[idx], np.shape(gt_xmax)[-1]])*gt_xmax
                gt_ymin = np.ones([self.fk[idx], self.fk[idx], np.shape(gt_ymin)[-1]])*gt_ymin
                gt_ymax = np.ones([self.fk[idx], self.fk[idx], np.shape(gt_ymax)[-1]])*gt_ymax

                # mask = tf.zeros([self.fk[idx], self.fk[idx], self.db_size[idx]])
                # label = tf.zeros([self.fk[idx], self.fk[idx], self.db_size[idx]*(class_num+4)])

                db_dim = default_box_dim[idx]
                db_xmin, db_ymin, db_xmax, db_ymax = np.split(db_dim, 4, axis=2)
                db_xmin = np.tile(db_xmin, [1, 1, size[batch_idx]])
                db_ymin = np.tile(db_ymin, [1, 1, size[batch_idx]])
                db_xmax = np.tile(db_xmax, [1, 1, size[batch_idx]])
                db_ymax = np.tile(db_ymax, [1, 1, size[batch_idx]])
                zero_tensor = np.zeros([self.fk[idx], self.fk[idx],
                    self.db_size[idx]*size[batch_idx]])
                # print(np.maximum(gt_xmin, db_xmin))
                # print(gt_xmax)
                # print(db_xmax)
                db_overlap = np.maximum(zero_tensor, np.multiply(
                            np.maximum(zero_tensor, np.minimum(gt_xmax, db_xmax) - np.maximum(gt_xmin, db_xmin)),
                            np.maximum(zero_tensor, np.minimum(gt_ymax, db_ymax) - np.maximum(gt_ymin, db_ymin)))) 
                db_overlap_loc = np.reshape(db_overlap, [self.fk[idx], self.fk[idx], 
                    self.db_size[idx], size[batch_idx]])
                db_union = np.multiply(gt_xmax-gt_xmin, gt_ymax-gt_ymin)+\
                    np.multiply(db_xmax-db_xmin, db_ymax-db_ymin) -db_overlap
                db_union_loc = np.reshape(db_union, [self.fk[idx], self.fk[idx], 
                    self.db_size[idx], size[batch_idx]])
                db_ratio = np.divide(db_overlap_loc, db_union_loc)
                #db_overlap_loc_ = db_ratio
                db_overlap_loc_label = np.argmax(db_ratio,axis=3).astype(np.int32) # eg. 38*38*4*1
                db_overlap_loc = np.amax(db_ratio, axis=3)
                db_overlap_loc_mask = np.greater(db_overlap_loc, 0.5).astype(np.float32)
                label_map = np.multiply(db_overlap_loc_mask ,
                    np.take(label_list, db_overlap_loc_label))
                label_one_hot = np.zeros(np.concatenate([[21],np.shape(label_map)]))
                # label_one_hot = tf.one_hot(cast(label_map, dtype=tf.int32), depth=21, on_value=1.0, off_value=0.0)
                # label_xmin = np.multiply(db_overlap_loc_mask, np.take(xmin_list, db_overlap_loc_label))
                # label_xmax = np.multiply(db_overlap_loc_mask, np.take(xmax_list, db_overlap_loc_label))
                # label_ymin = np.multiply(db_overlap_loc_mask, np.take(ymin_list, db_overlap_loc_label))
                # label_xmin = np.multiply(db_overlap_loc_mask, np.take(xmin_list, db_overlap_loc_label))
                label_xmin = np.take(xmin_list, db_overlap_loc_label)
                label_xmax = np.take(xmax_list, db_overlap_loc_label)
                label_ymin = np.take(ymin_list, db_overlap_loc_label)
                label_ymax = np.take(ymax_list, db_overlap_loc_label)
                feature_label = np.concatenate([
                    np.reshape(label_one_hot, [self.fk[idx], self.fk[idx], -1]),
                    np.reshape(label_xmin, [self.fk[idx], self.fk[idx], -1]),
                    np.reshape(label_xmax, [self.fk[idx], self.fk[idx], -1]),
                    np.reshape(label_ymin, [self.fk[idx], self.fk[idx], -1]),
                    np.reshape(label_ymax, [self.fk[idx], self.fk[idx], -1])], -1)
                feature_map_label[idx].append(feature_label)
                feature_map_mask[idx].append(db_overlap_loc_mask)
                pos_num.append(np.sum(db_overlap_loc_mask))
            pos_num_batch.append(np.sum(np.array(pos_num)))
        pos_num_batch = np.stack(pos_num_batch, 0)
        feature_map_label_batch = []    
        feature_map_mask_batch = []
        for idx in range(6):
            feature_map_label_batch.append(np.stack(feature_map_label[idx], 0))                  
            feature_map_mask_batch.append(np.stack(feature_map_mask[idx], 0))
        # print(pos_num_batch)
        return feature_map_label_batch, feature_map_mask_batch, pos_num_batch    

    """
    def encodeBboxTensor(self, xmin, ymin, xmax, ymax, label, size):
        label_shape = tf.concat([tf.expand_dims([-1]*kBatchSize, 0), 
            tf.expand_dims(tf.cast(size, dtype=tf.int32), 0)], 0)
        label_shape = tf.transpose(label_shape)
        feature_map_label = [[],[],[],[],[],[]]
        feature_map_mask = [[],[],[],[],[],[]]
        pos_num_batch = []
        for batch_idx in range(kBatchSize): 
            # xmin_list = tf.cast(tf.reshape(tf.decode_raw(xmin, tf.float32), label_shape), tf.float32)
            # xmax_list = tf.cast(tf.reshape(xmax, size), tf.float32)
            # ymin_list = tf.cast(tf.reshape(ymin, size), tf.float32)
            # ymax_list = tf.cast(tf.reshape(ymax, size), tf.float32)
            xmin_list = tf.reshape(tf.decode_raw(xmin, tf.float32), tf.cast(
                tf.expand_dims(size[batch_idx],0), dtype=tf.int32))
            xmax_list = tf.reshape(tf.decode_raw(xmax, tf.float32), tf.cast(
                tf.expand_dims(size[batch_idx],0), dtype=tf.int32)) 
            ymin_list = tf.reshape(tf.decode_raw(ymin, tf.float32), tf.cast(
                tf.expand_dims(size[batch_idx],0), dtype=tf.int32))
            ymax_list = tf.reshape(tf.decode_raw(ymax, tf.float32), tf.cast(
                tf.expand_dims(size[batch_idx],0), dtype=tf.int32))
            label_list = tf.reshape(tf.decode_raw(label, tf.float32), tf.cast(
                tf.expand_dims(size[batch_idx],0), dtype=tf.int32))
            # label_list = tf.cast(tf.reshape(label, size), tf.float32)
            pos_num = []
            default_box_dim = self.calculateDefaultBoxParam()
            for idx in range(6):
                gt_xmin = tf.reshape(tf.stack([xmin_list]*self.db_size[idx], axis = 1), [-1])
                gt_xmax = tf.reshape(tf.stack([xmax_list]*self.db_size[idx], axis = 1), [-1])
                gt_ymin = tf.reshape(tf.stack([ymin_list]*self.db_size[idx], axis = 1), [-1])
                gt_ymax = tf.reshape(tf.stack([ymax_list]*self.db_size[idx], axis = 1), [-1])

                gt_xmin = tf.ones([self.fk[idx], self.fk[idx], tf.shape(gt_xmin)[-1]])*gt_xmin
                gt_xmax = tf.ones([self.fk[idx], self.fk[idx], tf.shape(gt_xmax)[-1]])*gt_xmax
                gt_ymin = tf.ones([self.fk[idx], self.fk[idx], tf.shape(gt_ymin)[-1]])*gt_ymin
                gt_ymax = tf.ones([self.fk[idx], self.fk[idx], tf.shape(gt_ymax)[-1]])*gt_ymax

                # mask = tf.zeros([self.fk[idx], self.fk[idx], self.db_size[idx]])
                # label = tf.zeros([self.fk[idx], self.fk[idx], self.db_size[idx]*(class_num+4)])

                db_dim = tf.constant(default_box_dim[idx])
                db_xmin, db_xmax, db_ymin, db_ymax = tf.split(db_dim, 4, axis=2)
                db_xmin = tf.tile(db_xmin, tf.concat([[1], [1], tf.cast(
                    [size[batch_idx]], dtype=tf.int32)], 0))
                db_xmax = tf.tile(db_xmax, tf.concat([[1], [1], tf.cast(
                    [size[batch_idx]], dtype=tf.int32)], 0))
                db_ymin = tf.tile(db_ymin, tf.concat([[1], [1], tf.cast(
                    [size[batch_idx]], dtype=tf.int32)], 0))
                db_ymax = tf.tile(db_ymax, tf.concat([[1], [1], tf.cast(
                    [size[batch_idx]], dtype=tf.int32)], 0))         
                zero_tensor = tf.zeros(tf.stack([self.fk[idx], self.fk[idx], 
                    tf.cast(self.db_size[idx]*size[batch_idx], tf.int32)]))
                db_overlap = tf.maximum(zero_tensor, tf.multiply(
                            (tf.maximum(gt_xmin, db_xmin) - tf.minimum(gt_xmax, db_xmax)),
                            (tf.maximum(gt_ymin, db_ymin) - tf.minimum(gt_ymax, db_ymax)))) 
                db_overlap_loc = tf.reshape(db_overlap, tf.concat([
                    tf.convert_to_tensor([self.fk[idx], self.fk[idx], self.db_size[idx]]) 
                    ,tf.cast([size[batch_idx]], dtype=tf.int32)], 0))
                db_overlap_loc_ = db_overlap_loc
                db_index = tf.argmax(db_overlap_loc,axis=3)
                db_overlap_loc_label = tf.cast(tf.argmax(db_overlap_loc,axis=3), dtype=tf.int32) # eg. 38*38*4*1
                db_overlap_loc = tf.reduce_max(db_overlap_loc, axis=3)
                db_overlap_loc_mask = tf.cast(tf.greater(db_overlap_loc, 0.5), tf.float32)

                label_map = tf.multiply(db_overlap_loc_mask ,tf.cast(
                    tf.gather(label_list, db_overlap_loc_label), tf.float32))
                label_one_hot = tf.one_hot(tf.cast(label_map, dtype=tf.int32), depth=21, on_value=1.0, off_value=0.0)
                label_xmin = tf.multiply(db_overlap_loc_mask, tf.gather(xmin_list, db_overlap_loc_label))
                label_xmax = tf.multiply(db_overlap_loc_mask, tf.gather(xmax_list, db_overlap_loc_label))
                label_ymin = tf.multiply(db_overlap_loc_mask, tf.gather(ymin_list, db_overlap_loc_label))
                label_ymax = tf.multiply(db_overlap_loc_mask, tf.gather(ymax_list, db_overlap_loc_label))
                feature_label = tf.concat([
                    tf.reshape(label_one_hot, tf.cast(
                    [self.fk[idx], self.fk[idx], -1], tf.int32)),
                    tf.reshape(label_xmin, tf.cast(
                    [self.fk[idx], self.fk[idx], -1], tf.int32)),
                    tf.reshape(label_xmax, tf.cast(
                    [self.fk[idx], self.fk[idx], -1], tf.int32)),
                    tf.reshape(label_ymin, tf.cast(
                    [self.fk[idx], self.fk[idx], -1], tf.int32)),
                    tf.reshape(label_ymax, tf.cast(
                    [self.fk[idx], self.fk[idx], -1], tf.int32))], -1)
                feature_map_label[idx].append(feature_label)
                feature_map_mask[idx].append(db_overlap_loc_mask)
                pos_num.append(tf.reduce_sum(db_overlap_loc_mask))
            pos_num_batch.append(tf.reduce_sum(pos_num))
        pos_num_batch = tf.stack(pos_num_batch, 0)
        feature_map_label_batch = []    
        feature_map_mask_batch = []
        for idx in range(6):
            feature_map_label_batch.append(tf.stack(feature_map_label[idx], 0))                  
            feature_map_mask_batch.append(tf.stack(feature_map_mask[idx], 0))
        return feature_map_label_batch, feature_map_mask_batch, pos_num_batch
    """
    """
    def encodeBbox(self, bbox, label):
        #encoded_result = []
        #feature_map_label = []
        feature_map_label_0 = np.zeros(shape=[1, 38, 38, 4*(class_num+4)], dtype=np.float))
        feature_map_label.append(np.zeros(shape=[1, 19, 19, 6*(class_num+4)], dtype=np.float))
        feature_map_label.append(np.zeros(shape=[1, 10, 10, 6*(class_num+4)], dtype=np.float))
        feature_map_label.append(np.zeros(shape=[1, 5, 5, 6*(0+4)], dtype=np.float))
        feature_map_label.append(np.zeros(shape=[1, 3, 3, 6*(class_num+4)], dtype=np.float))
        feature_map_label.append(np.zeros(shape=[1, 1, 1, 6*(class_num+4)], dtype=np.float))


        feature_map_mask = []
        feature_map_mask.append(np.zeros(shape=[1, 38, 38, 4], dtype=np.float))
        feature_map_mask.append(np.zeros(shape=[1, 19, 19, 6], dtype=np.float))
        feature_map_mask.append(np.zeros(shape=[1, 10, 10, 6], dtype=np.float))
        feature_map_mask.append(np.zeros(shape=[1, 5, 5, 6], dtype=np.float))
        feature_map_mask.append(np.zeros(shape=[1, 3, 3, 6], dtype=np.float))
        feature_map_mask.append(np.zeros(shape=[1, 1, 1, 6], dtype=np.float))

        pos_box_num = 0
        for i in range(len(label)):
            result_box = self.detectOverlap(bbox[i])
            for box in result_box:
                l, i, j, db_idx = box
                x_min, y_min, x_max, y_max = bbox[i]
                cx = (x_min+x_max) / 2.0
                cy = (y_min+y_max) / 2.0
                w = x_max - x_min
                h = y_max - y_min
                feature_map_label[l, i, j, db_idx*feature_map_mask[l,3]+label] = 1.0
                feature_map_label[l, i, j, (class_num+0)*feature_map_mask[l,3]+db_idx] = cx
                feature_map_label[l, i, j, (class_num+1)*feature_map_mask[l,3]+db_idx] = cy
                feature_map_label[l, i, j, (class_num+2)*feature_map_mask[l,3]+db_idx] = w
                feature_map_label[l, i, j, (class_num+3)*feature_map_mask[l,3]+db_idx] = h
                feature_map_mask[l,i,j,db_idx] = 1.0
                pos_box_num += 1
        return feature_map_label, feature_map_mask, pos_box_num
    """
    def train(self):
        with tf.Session() as sess:
            with QueueRunners(sess):
                utils.initial_ops(sess, checkpoint_file)
                for train_step in range(1,kTotalSteps):
                    image, label, xmin, xmax, ymin, ymax, size = sess.run([self.train_set[0],
                        self.train_set[1], self.train_set[2],
                        self.train_set[3], self.train_set[4],
                        self.train_set[5], self.train_set[6]])
                    #bbox = self.buildBbox(xmin, xmax, ymin, ymax, size)

                    feature_map_label, feature_map_mask, pos_box_num = self.encodeBboxTensor(
                        xmin, ymin, xmax, ymax, label, size)
                    loss, assert_op= sess.run([self.train_ssd.train_op, self.train_ssd.assert_op], feed_dict={
                        self.train_ssd.image: image,
                        self.train_ssd.feature_map_label[0]: feature_map_label[0],
                        self.train_ssd.feature_map_label[1]: feature_map_label[1],
                        self.train_ssd.feature_map_label[2]: feature_map_label[2],
                        self.train_ssd.feature_map_label[3]: feature_map_label[3],
                        self.train_ssd.feature_map_label[4]: feature_map_label[4],
                        self.train_ssd.feature_map_label[5]: feature_map_label[5],
                        self.train_ssd.pos_mask[0]: feature_map_mask[0],
                        self.train_ssd.pos_mask[1]: feature_map_mask[1],
                        self.train_ssd.pos_mask[2]: feature_map_mask[2],
                        self.train_ssd.pos_mask[3]: feature_map_mask[3],
                        self.train_ssd.pos_mask[4]: feature_map_mask[4],
                        self.train_ssd.pos_mask[5]: feature_map_mask[5],
                        self.train_ssd.pos_box_num: pos_box_num,
                        self.train_ssd.step: train_step})
                    print("loss: %f"%(loss))
    