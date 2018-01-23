import tensorflow as tf
import numpy as np


class lap_pyramid():

    def __init__(self, k, image_size, kernel=np.reshape([[0.0025, 0.0125, 0.0200, 0.0125, 0.0025],
                                                         [0.0125, 0.0625, 0.1000, 0.0625, 0.0125],
                                                         [0.0200, 0.1000, 0.1600, 0.1000, 0.0200],
                                                         [0.0125, 0.0625, 0.1000, 0.0625, 0.0125],
                                                         [0.0025, 0.0125, 0.0200, 0.0125, 0.0025]],
                                                        (5, 5, 1, 1)),
                 strides=[1, 2, 2, 1], padding='VALID', loss=False):
        self.k = k
        self.kernel = kernel
        self.loss = loss
        self.strides = strides
        self.padding = padding
        self.image_size = image_size
        self.dn_filts, self.sigmas = self.DN_filters()
        if not self.loss:
            self.build_model()

    # Self pad to prevent a lot of zero padding - also using 'SYMMETRIC'
    def pad(self, image, pad=1, method="CONSTANT"):
        return tf.pad(image, [[0, 0], [pad, pad], [pad, pad], [0, 0]], method)

    def DN_filters(self):
        # These parameters were learned using the McGill dataset
        # Training_NLP_param.m
        sigmas = [0.0248, 0.0185, 0.0179, 0.0191, 0.0220, 0.2782]
        dn_filts = []

        dn_filts.append(np.reshape([[0, 0.1011, 0],
                                    [0.1493, 0, 0.1460],
                                    [0, 0.1015, 0.]],
                                   (3, 3, 1, 1)))

        dn_filts.append(np.reshape([[0, 0.0757, 0],
                                    [0.1986, 0, 0.1846],
                                    [0, 0.0837, 0]],
                                   (3, 3, 1, 1)))

        dn_filts.append(np.reshape([[0, 0.0477, 0],
                                    [0.2138, 0, 0.2243],
                                    [0, 0.0467, 0]],
                                   (3, 3, 1, 1)))

        dn_filts.append(np.reshape([[0, 0, 0],
                                    [0.2503, 0, 0.2616],
                                    [0, 0, 0]],
                                   (3, 3, 1, 1)))

        dn_filts.append(np.reshape([[0, 0, 0],
                                    [0.2598, 0, 0.2552],
                                    [0, 0, 0]],
                                   (3, 3, 1, 1)))

        dn_filts.append(np.reshape([[0, 0, 0],
                                    [0.2215, 0, 0.0717],
                                    [0, 0, 0]],
                                   (3, 3, 1, 1)))

        return dn_filts, sigmas

    def normalise(self, convs):
        norm = []
        for i in range(0, len(convs)):
            n = tf.nn.conv2d(tf.abs(convs[i]), self.dn_filts[i], strides=[1, 1, 1, 1], padding="SAME")
            norm.append(convs[i] / (self.sigmas[i] + n))
        return norm

    def convs(self):
        J = self.im
        pyr = []
        for i in range(0, self.k - 1):
            I = tf.nn.conv2d(self.pad(J, pad=2), self.kernel, strides=self.strides, padding=self.padding)
            I_up = tf.image.resize_images(I, [int(np.ceil(self.image_size[0] / (2 ** i))),
                                              int(np.ceil(self.image_size[1] / (2 ** i)))])
            I_up_conv = tf.nn.conv2d(self.pad(I_up, pad=2), self.kernel, strides=[1, 1, 1, 1], padding=self.padding)
            pyr.append(J - I_up_conv)
            J = I
        pyr.append(J)
        return self.normalise(pyr)

    def convs_loss_function(self, im):
        J = im
        pyr = []
        for i in range(0, self.k - 1):
            I = tf.nn.conv2d(self.pad(J, pad=2), self.kernel, strides=self.strides, padding=self.padding)
            I_up = tf.image.resize_images(I, [int(np.ceil(self.image_size[0] / (2 ** i))),
                                              int(np.ceil(self.image_size[1] / (2 ** i)))])
            I_up_conv = tf.nn.conv2d(self.pad(I_up, pad=2), self.kernel, strides=[1, 1, 1, 1], padding=self.padding)
            pyr.append(J - I_up_conv)
            J = I
        pyr.append(J)
        return self.normalise(pyr)

    def loss_function(self, real, pred):
        realpyr = self.convs_loss_function(real)
        predpyr = self.convs_loss_function(pred)
        total = tf.convert_to_tensor(0, dtype=tf.float32)
        for i in range(0, self.k):
            total = tf.add(total, tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(realpyr[i], predpyr[i])))))
        return tf.divide(total, tf.convert_to_tensor(self.k, dtype=tf.float32))

    def forward(self, image):
        convs_out, convs_up_out = self.sess.run([self.convs_, self.convs_up_],
                                                feed_dict={self.im: image})
        return convs_out, convs_up_out

    def compare(self, image1, image2):
        self.image_size = image1.shape
        convs_up_out1 = self.sess.run(self.convs_up_, feed_dict={self.im: image1})
        convs_up_out2 = self.sess.run(self.convs_up_, feed_dict={self.im: image2})
        convs_up_out1r = [np.squeeze(x) for x in convs_up_out1]
        convs_up_out2r = [np.squeeze(x) for x in convs_up_out2]
        rmse = []
        for i in range(0, self.k):
            rmse.append(np.sqrt(np.mean((convs_up_out1[i] - convs_up_out2[i]) ** 2)))
        return np.mean(rmse)

    def output_pyramid(self, im):
        convs = self.sess.run(self.convs_up_, feed_dict={self.im: im})
        convs = [np.squeeze(x) for x in convs]
        return convs

    def build_model(self):
        self.im = tf.placeholder('float32', [None, None, None, 1])
        self.convs_up_ = self.convs()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
