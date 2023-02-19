#!/Yan/miniforge3/envs/
# -*- coding:utf-8 -*-

import tensorflow as tf


class Sampling(tf.keras.layers.Layer):
    """Re-parameterization Layer: Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))  # re-parameterization
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon  # equal to sample with the mean z and sd sqrt(var z)


class Encoder(tf.keras.Model):
    """Maps image to a triplet (z_mean, z_log_var, z)."""

    def __init__(self, name='COV_encoder', **kwargs):
        super(Encoder, self).__init__(name=name)
        self.dropout = tf.keras.layers.Dropout(rate=0.3)  # Dropout Layer
        self.conv_1 = tf.keras.layers.Conv2D(
            filters=32,  # num of kernel
            kernel_size=[5, 5],
            strides=2,
            padding='same',  # valid or same
            activation='relu'
        )
        # self.bn_1 = tf.keras.layers.BatchNormalization()
        # self.relu = tf.keras.layers.Activation('relu')  # applying activation after BN!

        self.conv_2 = tf.keras.layers.Conv2D(
            filters=64,  # num of kernel
            kernel_size=[5, 5],
            strides=2,
            padding='same',  # valid or same
            activation='relu'
        )
        self.conv_3 = tf.keras.layers.Conv2D(
            filters=64,  # num of kernel
            kernel_size=[5, 5],
            padding='same',  # valid or same
            activation='relu'
        )
        # self.bn_2 = tf.keras.layers.BatchNormalization()

        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(units=150)
        # self.bn_3 = tf.keras.layers.BatchNormalization()

        self.dense_mean = tf.keras.layers.Dense(**kwargs)  # define the size of latent space
        self.dense_log_var = tf.keras.layers.Dense(**kwargs)
        self.sampling = Sampling()

    @tf.function
    def call(self, inputs, training=None):
        x = self.conv_1(inputs)
        # x = self.bn_1(x, training=training)
        # x = self.relu(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        # x = self.bn_2(x, training=training)
        # x = self.relu(x)
        # x = self.dropout(x, training=training)
        x = self.flatten(x)
        x = self.dropout(x, training=training)
        x = self.dense(x)
        x = self.dropout(x, training=training)
        # x = self.bn_3(x, training=training)
        # x = self.relu(x)
        # x = self.dropout(x, training=training)
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z


class Decoder(tf.keras.Model):
    """Converts z, the encoded digit vector, back into a readable digit."""

    def __init__(self, name='COV_decoder', **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        # self.dropout = tf.keras.layers.AlphaDropout(rate=0.3)
        self.dense_1 = tf.keras.layers.Dense(150)
        # self.bn_1 = tf.keras.layers.BatchNormalization()
        # self.relu = tf.keras.layers.Activation('relu')  # applying activation after BN!

        self.dense_2 = tf.keras.layers.Dense(10 * 10 * 64)
        # self.bn_2 = tf.keras.layers.BatchNormalization()
        self.reshape = tf.keras.layers.Reshape((10, 10, 64))

        self.Tconv_1 = tf.keras.layers.Conv2DTranspose(
            filters=64, kernel_size=[5, 5], activation='relu', strides=2, padding="same")
        # self.bn_3 = tf.keras.layers.BatchNormalization()
        self.Tconv_2 = tf.keras.layers.Conv2DTranspose(
            filters=32, kernel_size=[5, 5], activation='relu', strides=2, padding="same")
        # self.bn_4 = tf.keras.layers.BatchNormalization()
        self.Tconv_3 = tf.keras.layers.Conv2DTranspose(
            filters=1, kernel_size=[5, 5], activation="sigmoid", padding="same")
        self.dense_category = tf.keras.layers.Dense(7)    # for category loss

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 3], dtype=tf.float32)])
    def call(self, inputs):
        y = self.dense_category(inputs)  # for category loss
        output_category = tf.nn.softmax(y)  # for category loss
        x = self.dense_1(inputs)
        # x = self.bn_1(x, training=training)
        # x = self.relu(x)
        x = self.dense_2(x)
        # x = self.bn_2(x, training=training)
        # x = self.relu(x)
        # x = self.dropout(x, training=training)
        x = self.reshape(x)
        x = self.Tconv_1(x)
        # x = self.bn_3(x, training=training)
        # x = self.relu(x)
        # x = self.dropout(x, training=training)
        x = self.Tconv_2(x)
        # x = self.bn_4(x, training=training)
        # x = self.relu(x)
        # x = self.dropout(x, training=training)
        output = self.Tconv_3(x)
        return output, output_category


class CVAE(tf.keras.Model):
    """Combines the encoder and decoder into an end-to-end model for training."""

    def __init__(self, name='Conv_VAE', **kwargs):
        super(CVAE, self).__init__(name=name)
        self.encoder = Encoder(**kwargs)
        self.decoder = Decoder()

    def call(self, inputs, training=None):
        z_mean, z_log_var, z = self.encoder(inputs, training=training)
        reconstructed = self.decoder(z)
        # Add KL divergence regularization loss.
        kl_loss = - 0.5 * tf.reduce_sum(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
        self.add_loss(kl_loss)  # add KL divergence regularization loss to self.losses
        # add_loss attribute define some losses that may be dependent on the inputs passed when calling a
        # layer instead of barely output of the whole model
        return reconstructed



