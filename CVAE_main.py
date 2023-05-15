#!/Yan/miniforge3/envs/
# -*- coding:utf-8 -*-
# Variational de-noised Auto-Encoder Training


import tensorflow as tf
from . import CVAE_model
from . import Imageset
# import os
# import warnings
import numpy as np
import matplotlib.pyplot as plt


# warnings.filterwarnings("ignore")  # ignore the matching warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # shield the log info except fatal info

class CVAE:
    """Convolutional Auto-encoder training, testing and visualizing"""

    font = {"color": "darkred",  # visualization font
            "size": 13,
            "family": "serif"}

    def __init__(self, mode='new', n_dim=3, model_path='./ModelSaved'):
        # set up the model (with latent space as n-dimensional) and dataset
        if mode == 'new':
            self.model = CVAE_model.CVAE(units=n_dim)  # call a model
            self.mode = 'new'
            self.n_dim = n_dim
        elif mode == 'model_saved':
            self.model = tf.saved_model.load(model_path)
            self.mode = 'model_saved'
        else:
            raise Exception('mode should be new or model_saved')

        self.train_file_set = None
        self.train_dataset = None

        self.test_file_set = None
        self.test_dataset = None

        self.loss_metric = None  # for loss calculation and optimization
        self.optimizer = None


    def checkpoint_load(self, log_path, n):
        checkpoint = tf.train.Checkpoint(myAwesomeModel=self.model)
        # checkpoint.restore(tf.train.latest_checkpoint(Logs_path))
        checkpoint.restore(log_path + '/model.ckpt-' + str(n))

    def cafe_load(self, train_dir, test_dir, size_height=40, size_length=40, normalization=0, **kwargs):
        self.train_file_set = Imageset.CAFE(train_dir, '.jpg')
        self.train_dataset = self.train_file_set.get_dataset(self.train_file_set.filenames_dataset,
                                                             Imageset.CAFE._decode_and_resize, size_height,
                                                             size_length, normalization=normalization)

        self.test_file_set = Imageset.CAFE(test_dir, '.jpg')
        self.test_dataset = self.test_file_set.get_dataset(self.test_file_set.filenames_dataset,
                                                           Imageset.CAFE._decode_and_resize, size_height,
                                                           size_length, normalization=normalization)
        # test_one_batch = self.test_dataset.batch(self.test_file_set.num_data)

    def jaffe_load(self, train_dir, test_dir, size_height=40, size_length=40, normalization=0, **kwargs):
        self.train_file_set = Imageset.JAFFE(train_dir, '.tiff')
        self.train_dataset = self.train_file_set.get_dataset(size_height, size_length, normalization=normalization)

        self.test_file_set = Imageset.JAFFE(test_dir, '.tiff')
        self.test_dataset = self.test_file_set.get_dataset(size_height, size_length, normalization=normalization)

    def fer2013_load(self, train_dir, test_dir, size_height=40, size_length=40, normalization=0, **kwargs):
        self.train_file_set = Imageset.FER2013(train_dir, '.jpg')
        self.train_dataset = self.train_file_set.get_dataset(size_height, size_length, normalization=normalization)

        self.test_file_set = Imageset.FER2013(test_dir, '.jpg')
        self.test_dataset = self.test_file_set.get_dataset(size_height, size_length, normalization=normalization)

    def cross_validate(self, fold, n_epoch=1, batch_size=100, learning_rate=0.001, left_size=0.9, **kwargs):
        """K-fold CV to find hyper-parameter"""

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.loss_metric = tf.keras.metrics.Mean()

        self.model = CVAE_model.CVAE(units=self.n_dim)
        train_set, valid_set = tf.keras.utils.split_dataset(self.train_dataset, left_size=left_size, shuffle=True)
        for epoch in range(n_epoch):
            epoch_loss = 0
            class_loss = 0
            trainings = train_set.batch(batch_size)
            num_batch = trainings.cardinality().numpy()
            for train_batch, Labels in trainings:
                train_batch = tf.cast(train_batch, tf.float32)
                batch_loss, cat_loss = self._train_vae(train_batch, Labels, **kwargs)
                epoch_loss += batch_loss
                class_loss += cat_loss

            print("TRAINING(fold:%02d): epoch: [%02d/%02d] Batch_Loss: %.4f Categorical_Loss: %.4f"
                  % (fold, epoch, n_epoch, epoch_loss/num_batch, class_loss/num_batch))

            validations = valid_set.batch(200)  # only one batch
            for data, label in validations:
                _, classification = self.model(data, training=False)
                cat_loss = tf.reduce_sum(
                    tf.keras.losses.categorical_crossentropy(y_true=label, y_pred=classification))
            print("VALIDATION(fold:%02d): epoch: [%02d/%02d] Categorical_Loss: %.4f"
                  % (fold, epoch, n_epoch, cat_loss))

    def train(self, save_path, n_epoch=1, batch_size=100, learning_rate=0.001, **kwargs):

        checkpoint = tf.train.Checkpoint(myAwesomeModel=self.model)  # get a checkpoint，object is our model
        manager = tf.train.CheckpointManager(checkpoint, directory=save_path,
                                             checkpoint_name='model.ckpt', max_to_keep=8)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.loss_metric = tf.keras.metrics.Mean()  # an object who calculates the mean value of a given matrix

        buffer_size = int(self.train_file_set.num_data / 3)  # buffer size when selecting training batch
        for epoch in range(n_epoch):
            trainings = self.train_dataset.shuffle(buffer_size=buffer_size).batch(batch_size).prefetch(
                tf.data.experimental.AUTOTUNE)
            for step, [train_batch, Labels] in enumerate(trainings):
                train_batch = tf.cast(train_batch, tf.float32)  # apply only if face_detect is on
                batch_loss, _ = self.train_vae(self.optimizer, self.loss_metric,
                                               self.model, train_batch, Labels, **kwargs)
                print("step: [%02d/%02d] Batch_Loss: %.4f" % (step, epoch, batch_loss))
            if (epoch + 1) % 50 == 0:  # save model every 50 epochs
                path = manager.save()
                print('model saved to %s' % path)
        print('Optimization Finished!')

    @classmethod
    @tf.function
    def train_vae(cls, optimizer, loss_metric, model, data, lab, weight_catloss=3):
        with tf.GradientTape() as tape:
            reconstruct = model(data, training=True)
            loss = tf.reduce_sum(tf.square(data - reconstruct[0]))
            loss += tf.reduce_sum(model.losses)  # Add Dkl regularization loss
            # Add a classification loss to make latent space more divisible
            cat_loss = tf.reduce_sum(tf.keras.losses.categorical_crossentropy(y_true=lab, y_pred=reconstruct[1]))
            loss += weight_catloss * cat_loss
        grads = tape.gradient(loss, model.variables)  # all variables are trainable
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
        loss_metric(loss)
        return loss_metric.result(), cat_loss

    def save(self, model_path="./ModelSaved", target='all', **kwargs):
        if target == 'all':
            tf.saved_model.save(self.model, model_path, **kwargs)
        elif target == 'encoder':
            tf.saved_model.save(self.model.encoder, model_path, **kwargs)
        elif target == 'decoder':
            tf.saved_model.save(self.model.decoder, model_path, **kwargs)
        else:
            ValueError('target should be all, decoder or encoder')

    @classmethod
    def reconstruct(cls, model, filename, dataset, n_show=10):
        # random visualization of the reconstruction
        rand_idx = np.random.randint(0, int(filename.num_data / n_show), 1)
        # randomly choose a batch to show
        img_show = dataset.batch(n_show)

        for index, [data, label] in enumerate(img_show):
            if index == rand_idx:
                data = tf.cast(data, tf.float32)
                latent_code = model.encoder(data, training=False)
                pred_img = model.decoder(latent_code[0])
                # plt.style.use("dark_background")
                f, a = plt.subplots(2, n_show)
                for i in range(n_show):
                    a[0][i].imshow(data.numpy()[i, :, :, :].squeeze())
                    a[1][i].imshow(pred_img[0].numpy()[i, :, :, :].squeeze())
                f.show()
                plt.suptitle('VAE_images_reconstructed', fontdict=cls.font)
                plt.draw()
                plt.show()

    @classmethod
    def latent_sample(cls, model, n_show=10, std=1):

        # randomly choose a batch to show
        img_show = tf.random.normal([10, 3], stddev=std)
        data = tf.cast(img_show, tf.float32)

        pred_img = model(data)
        # plt.style.use("dark_background")
        f, a = plt.subplots(1, n_show)
        for i in range(n_show):
            a[i].imshow(pred_img[0].numpy()[i, :, :, :].squeeze())
        f.show()
        plt.suptitle('VAE_images_sample in LS', fontdict=cls.font)
        plt.draw()
        plt.show()

    @classmethod
    def latent_space(cls, model, filename, dataset, latent_dims, label_list):
        """display how the latent space clusters different data points in an n-dimensional space"""
        x = np.random.normal(0, 1, [1, latent_dims])
        plot_batch = dataset.batch(batch_size=100)  # take the trainset as plot data

        for data, labels in plot_batch:  # calculate the latent codes
            data = tf.cast(data, tf.float32)  # crucial in loading a saved_model and calling it with test data
            latent_code = model(data, training=False)
            x = tf.concat([x, latent_code[0]], 0)  # [0]: z-mean
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        x = x.numpy()[1:, :]

        for key, value in label_list.items():
            ax.scatter(x[np.where(np.array(filename.labels) == int(key)), 0],
                       x[np.where(np.array(filename.labels) == int(key)), 1],
                       x[np.where(np.array(filename.labels) == int(key)), 2],
                       alpha=0.5, label=value)

        # plt.title("Data Points in Latent Space")
        plt.tight_layout()
        ax.legend()
        # plt.savefig('./figure/save.jpg')
        plt.show()

    @classmethod
    def manifold(cls, model, scale, z=0):
        """Grid plot for sampling 3-d latent space: x,y are firmed as [-3, 3], but z are sampled as input parameters.
        scale means how many points show"""
        dim = 6 * scale  # scale
        f, a = plt.subplots(dim, dim)
        plt.setp(a.flat, xticks=[], yticks=[])
        for i in range(dim):
            for j in range(dim):
                sample = model(tf.constant([[-3 + j / scale, 3 - i / scale, z]], dtype=tf.float32))
                a[i][j].imshow(sample[0].numpy()[0, :, :, 0], cmap='gray')
        # plt.suptitle('Grid Latent Space', fontdict=cls.font)
        f.subplots_adjust(wspace=0, hspace=0, bottom=0.15, right=0.88)
        plt.show()

    @classmethod
    def test(cls, model, filename, dataset):
        """display how the latent space clusters different data points in an n-dimensional space"""

        test_batch = dataset.batch(batch_size=len(dataset))  # take the trainset as plot data

        for data, labels in test_batch:  # calculate the latent codes
            data = tf.cast(data, tf.float32)  # crucial in loading a saved_model and calling it with test data
            _, predictions = model(data, training=False)

        predicts = np.argmax(predictions, axis=1)
        print('overall accurate:', np.sum(filename.labels == predicts)/len(predicts))

        # confusion matrix
        confusion_mat = np.zeros((len(np.unique(filename.labels)), len(np.unique(filename.labels))))
        for i in range(len(np.unique(filename.labels))):
            for j in range(len(np.unique(filename.labels))):
                confusion_mat[i, j] = np.sum(predicts[np.where(np.array(filename.labels) == i)] == j)

        print('confusion matrix:\n', confusion_mat)













