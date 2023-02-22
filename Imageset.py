#!/Yan/miniforge3/envs/
# -*- coding:utf-8 -*-
# For loading images dataset


import tensorflow as tf
import cv2
from mtcnn import MTCNN
import numpy as np
import os


def read_csv2list(filepath):
    """read label.csv and return a list"""

    file = open(filepath, 'r', encoding="utf-8")
    context = file.read()  # as str
    list_result = context.split("\n")[1:-1]
    length = len(list_result)
    for i in range(length):
        list_result[i] = int(list_result[i].split(",")[-1])
    file.close()  # file must be closed after manipulating
    return list_result


class CAFE:
    """import, decode and get batches from the image dataset using tf.io.read_file"""

    def __init__(self, data_dir, suffix):
        self.filenames = [data_dir + '/' + filename for filename in os.listdir(data_dir)
                          if os.path.splitext(filename)[1] == suffix]
        self.filenames.sort(key=lambda x: int(x.split('/')[-1][:-4]))  # data path list sorted by NO.
        self.filenames_tensor = tf.constant(self.filenames)  # convert list as tensor
        self.labels = read_csv2list(data_dir + '/' + 'label.csv')
        self.filenames_dataset = tf.data.Dataset.from_tensor_slices((self.filenames_tensor,
                                                                     tf.one_hot(self.labels, depth=7).numpy()))
        self.num_data = len(self.filenames)  # dim of data

        self.detector = None

        # get max and min value from each image ROI
        # self.max = []
        # self.min = []
        # for i in self.filenames_dataset:    # for the scaling
        #     image_string = tf.io.read_file(i)
        #     image_decode = tf.image.decode_jpeg(image_string)[15:-9, :]  # define the size we extract from images
        #     image_max = np.max(image_decode.numpy())
        #     self.max.append(image_max)
        #     image_min = np.min(image_decode.numpy())
        #     self.min.append(image_min)

    @staticmethod
    def get_dataset(filenames_dataset, _decode_and_resize, size_height, size_length, normalization=0):
        """dataset with fixed face-cropping window"""
        # define graph size and whether normalization
        dataset = filenames_dataset.map(
            map_func=lambda x, y: _decode_and_resize(x, y, size_height=size_height, size_length=size_length,
                                                     normalization=normalization),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # dataset = []
        # for index, i in enumerate(self.filenames_dataset):  # do the scaling
        #     image_string = tf.io.read_file(i)
        #     image_decode = tf.image.decode_jpeg(image_string)[15:-9, :]
        #     if normalization == 0:
        #         image_resized = (tf.image.resize(image_decode,
        #                                          [size_height, size_length]) - self.min[index]) / \
        #                         (self.max[index] - self.min[index])
        #     else:
        #         image_resized = tf.image.resize(image_decode, [size_height, size_length])
        #     dataset.append(image_resized)
        # dataset = tf.data.Dataset.from_tensor_slices((dataset, labels))
        return dataset

    def get_dataset_facedetection(self, size_height, size_length, normalization=0):
        """generate a dataset with face location automatically"""
        images = np.zeros((self.num_data, size_height, size_length, 1))
        self.detector = MTCNN()
        for i, filename in enumerate(self.filenames):
            images[i, :, :, :] = self._decode_and_facedetect(filename, size_height=size_height,
                                                             size_length=size_length, normalization=normalization)
        dataset = tf.data.Dataset.from_tensor_slices((images, tf.one_hot(self.labels, depth=7).numpy()))
        return dataset

    @staticmethod
    def _decode_and_resize(filename, labels, size_height, size_length, normalization=0):
        """normalization = 0 means the data has been normalized, otherwise not"""

        image_string = tf.io.read_file(filename)  # read image as string code
        # image_decode = tf.image.decode_jpeg(image_string)
        image_decode = tf.image.decode_jpeg(image_string)[15:-9, :]  # turn image string as array while extract the ROI
        if normalization == 0:
            image_resized = tf.image.resize(image_decode, [size_height, size_length]) / 255
        else:
            image_resized = tf.image.resize(image_decode, [size_height, size_length])
        # define the size and normalization
        return tf.reshape(image_resized, [size_height, size_length, 1]), labels

    # decode the img and detect the face area
    def _decode_and_facedetect(self, filename, size_height, size_length, normalization=0):
        """normalization = 0 means the data has been normalized, otherwise not; face detection"""

        image_decode = cv2.imread(filename)
        face = self.detector.detect_faces(image_decode)
        if len(face):
            x_0 = face[0]['box'][1] + int(1/5 * face[0]['box'][-1])
            x_1 = face[0]['box'][1] + int(6/7 * face[0]['box'][-1])
            y_0 = face[0]['box'][0]
            y_1 = face[0]['box'][0] + face[0]['box'][2]
            image_decode = image_decode[x_0:x_1, y_0:y_1, :]  # face detection
        else:
            image_decode = image_decode[15:-9, :, :]
        image_decode = 0.2989 * image_decode[:, :, 0] + \
                        0.5870 * image_decode[:, :, 1] + 0.1140 * image_decode[:, :, 2]
        if normalization == 0:
            image_resized = np.expand_dims(image_decode.astype(np.float32) / 255.0, axis=-1)
        else:
            image_resized = np.expand_dims(image_decode.astype(np.float32), axis=-1)
        image_resized = tf.image.resize(image_resized, [size_height, size_length])
        return image_resized


class JAFFE:
    """import, decode and get batches from the image dataset using tf.io.read_file"""

    def __init__(self, data_dir, suffix):
        self.filenames = [data_dir + '/' + filename for filename in os.listdir(data_dir)
                          if os.path.splitext(filename)[1] == suffix]
        self.filenames.sort(key=lambda x: int(x.split('/')[-1].split('.')[-2]))  # data path list sorted by NO.
        # self.filenames = tf.constant(self.filenames)  # convert list as tensor
        self.labels = read_csv2list(data_dir + '/' + 'label.csv')
        # self.filenames_dataset = tf.data.Dataset.from_tensor_slices((self.filenames,
        #                                                              tf.one_hot(self.labels, depth=7).numpy()))
        self.num_data = len(self.filenames)  # dim of data

        self.detector = None

        # get max and min value from each image ROI
        # self.max = []
        # self.min = []
        # for i in self.filenames_dataset:    # for the scaling
        #     image_string = tf.io.read_file(i)
        #     image_decode = tf.image.decode_jpeg(image_string)[15:-9, :]  # define the size we extract from images
        #     image_max = np.max(image_decode.numpy())
        #     self.max.append(image_max)
        #     image_min = np.min(image_decode.numpy())
        #     self.min.append(image_min)

    def get_dataset(self, size_height, size_length, normalization=0):
        # define graph size and whether normalization
        images = np.zeros((self.num_data, size_height, size_length, 1))
        # self.detector = MTCNN()
        for i, filename in enumerate(self.filenames):
            images[i, :, :, :] = self._decode_and_resize(filename, size_height=size_height, size_length=size_length,
                                                         normalization=normalization)

        dataset = tf.data.Dataset.from_tensor_slices((images, tf.one_hot(self.labels, depth=7).numpy()))
        return dataset

    # decode the img and normalization (for dataset.map function)
    def _decode_and_resize(self, filename, size_height, size_length, normalization=0):
        """normalization = 0 means the data has been normalized, otherwise not"""

        image_decode = cv2.imread(filename, 2)[98:216, 73:183]  # ROI
        image_decode = np.expand_dims(image_decode.astype(np.float32), axis=-1)
        # face = self.detector.detect_faces(image_decode)
        # xlim = [face[0]['box'][1] + int(face[0]['box'][3] * 1 / 5),
        # face[0]['box'][1] + int(face[0]['box'][3] * 6 / 7)]
        # ylim = [face[0]['box'][0], face[0]['box'][0] + face[0]['box'][2]]
        # image_decode = image_decode[xlim[0]:xlim[1], ylim[0]:ylim[1], :]  # face detection
        if normalization == 0:
            image_resized = tf.image.resize(image_decode, [size_height, size_length]) / 255.0
        else:
            image_resized = tf.image.resize(image_decode, [size_height, size_length])
        # define the size and normalization
        # image_resized = 0.2989 * image_resized[:, :, 0] + \
        #                 0.5870 * image_resized[:, :, 1] + 0.1140 * image_resized[:, :, 2]
        return image_resized


class FER2013:
    """import, decode and get batches from the image dataset using tf.io.read_file"""

    def __init__(self, data_dir, suffix):
        self.filenames_angry = tf.constant([data_dir + '/angry/' + filename for filename in os.listdir(
            data_dir + '/angry/') if os.path.splitext(filename)[1] == suffix])
        self.filenames_disgust = tf.constant([data_dir + '/disgust/' + filename for filename in os.listdir(
            data_dir + '/disgust/') if os.path.splitext(filename)[1] == suffix])
        self.filenames_fear = tf.constant([data_dir + '/fear/' + filename for filename in os.listdir(
            data_dir + '/fear/') if os.path.splitext(filename)[1] == suffix])
        self.filenames_happy = tf.constant([data_dir + '/happy/' + filename for filename in os.listdir(
            data_dir + '/happy/') if os.path.splitext(filename)[1] == suffix])
        self.filenames_neutral = tf.constant([data_dir + '/neutral/' + filename for filename in os.listdir(
            data_dir + '/neutral/') if os.path.splitext(filename)[1] == suffix])
        self.filenames_sad = tf.constant([data_dir + '/sad/' + filename for filename in os.listdir(
            data_dir + '/sad/') if os.path.splitext(filename)[1] == suffix])
        self.filenames_surprise = tf.constant([data_dir + '/surprise/' + filename for filename in os.listdir(
            data_dir + '/surprise/') if os.path.splitext(filename)[1] == suffix])
        self.filenames = tf.concat([self.filenames_angry, self.filenames_disgust, self.filenames_fear,
                                    self.filenames_happy, self.filenames_neutral, self.filenames_sad,
                                    self.filenames_surprise], axis=0)  # concat all images
        self.labels = tf.concat([tf.zeros(self.filenames_angry.shape, dtype=tf.int32),
                                 tf.ones(self.filenames_disgust.shape, dtype=tf.int32),
                                 (tf.ones(self.filenames_fear.shape, dtype=tf.int32) * 2),
                                 (tf.ones(self.filenames_happy.shape, dtype=tf.int32) * 3),
                                 (tf.ones(self.filenames_neutral.shape, dtype=tf.int32) * 4),
                                 (tf.ones(self.filenames_sad.shape, dtype=tf.int32) * 5),
                                 (tf.ones(self.filenames_surprise.shape, dtype=tf.int32) * 6)],
                                axis=0)
        self.filenames_dataset = tf.data.Dataset.from_tensor_slices(
            (self.filenames, tf.one_hot(self.labels, depth=7)))
        self.num_data = self.filenames.shape[0]  # dim of data

        # get max and min value from each image ROI
        # self.max = []
        # self.min = []
        # for i in self.filenames_dataset:    # for the scaling
        #     image_string = tf.io.read_file(i)
        #     image_decode = tf.image.decode_jpeg(image_string)[15:-9, :]  # define the size we extract from images
        #     image_max = np.max(image_decode.numpy())
        #     self.max.append(image_max)
        #     image_min = np.min(image_decode.numpy())
        #     self.min.append(image_min)

    def get_dataset(self, size_height, size_length, normalization=0):
        # define graph size and whether normalization
        dataset = self.filenames_dataset.map(
            map_func=lambda x, y: self._decode_and_resize(x, y, size_height=size_height,
                                                          size_length=size_length, normalization=normalization),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # dataset = []
        # for index, i in enumerate(self.filenames_dataset):  # do the scaling
        #     image_string = tf.io.read_file(i)
        #     image_decode = tf.image.decode_jpeg(image_string)[15:-9, :]
        #     if normalization == 0:
        #         image_resized = (tf.image.resize(image_decode,
        #                                          [size_height, size_length]) - self.min[index]) / \
        #                         (self.max[index] - self.min[index])
        #     else:
        #         image_resized = tf.image.resize(image_decode, [size_height, size_length])
        #     dataset.append(image_resized)
        # dataset = tf.data.Dataset.from_tensor_slices((dataset, labels))
        return dataset

    def _decode_and_resize(self, filename, labels, size_height, size_length, normalization=0):
        """normalization = 0 means the data has been normalized, otherwise not"""

        image_string = tf.io.read_file(filename)  # read image as string code
        image_decode = tf.image.decode_jpeg(image_string)  # turn image string as array while extract the ROI
        if normalization == 0:
            image_resized = tf.image.resize(image_decode, [size_height, size_length]) / 255
        else:
            image_resized = tf.image.resize(image_decode, [size_height, size_length])
        # define the size and normalization
        return tf.reshape(image_resized, [size_height, size_length, 1]), labels

