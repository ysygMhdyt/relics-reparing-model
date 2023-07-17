import tensorflow as tf
import keras
import keras.layers as layers

alpha_Leaky = 0.2

class Generator(keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        # 定义要用到的层
        # 卷积层
        self.conv1 = layers.Conv2D(64, 5, strides=2, padding='SAME', activation=None)
        self.conv2 = layers.Conv2D(128, 5, strides=2, padding='SAME', activation=None)
        self.conv3 = layers.Conv2D(256, 5, strides=2, padding='SAME', activation=None)
        self.conv4 = layers.Conv2D(512, 5, strides=2, padding='SAME', activation=None)
        # LeakyReLU激活层
        self.activation_Leaky = layers.LeakyReLU(alpha=alpha_Leaky)
        # ReLU激活层
        self.activation_Relu = layers.Activation(activation='relu')
        # 批量归一化层：
        # 用于卷积
        self.bn1 = layers.BatchNormalization(scale=False)   # input:_*_*128
        self.bn2 = layers.BatchNormalization(scale=False)   # input:256
        self.bn3 = layers.BatchNormalization(scale=False)   # input:512
        # 用于转置卷积
        self.bn1_tr = layers.BatchNormalization(scale=False)    # input 512
        self.bn2_tr = layers.BatchNormalization(scale=False)    # input 256
        self.bn3_tr = layers.BatchNormalization(scale=False)    # input 128
        self.bn4_tr = layers.BatchNormalization(scale=False)    # input 64
        # 全连接层
        self.fc = layers.Dense(units=512, activation=None)
        # 转置卷积层
        self.trans_conv1 = layers.Conv2DTranspose(256, 5, strides=2, padding='SAME', activation=None)
        self.trans_conv2 = layers.Conv2DTranspose(128, 5, strides=2, padding='SAME', activation=None)
        self.trans_conv3 = layers.Conv2DTranspose(64, 5, strides=2, padding='SAME', activation=None)
        self.trans_conv4 = layers.Conv2DTranspose(3, 5, strides=2, padding='SAME', activation=None)
        # 输出层，使用tanh激活
        self.out = layers.Activation(activation='tanh')

    def call(self, inputs, training=True, mask=None):
        # 卷积
        conv1 = self.conv1(inputs)  # 64
        activation1 = self.activation_Leaky(conv1)

        conv2 = self.conv2(activation1)  # 128
        bn1 = self.bn1(conv2, training=training)
        activation2 = self.activation_Leaky(bn1)

        conv3 = self.conv3(activation2)  # 256
        bn2 = self.bn2(conv3, training=training)
        activation3 = self.activation_Leaky(bn2)

        conv4 = self.conv4(activation3)  # 512
        bn3 = self.bn3(conv4, training=training)
        activation4 = self.activation_Leaky(bn3)

        # 全连接
        fc = self.fc(activation4)
        fc_reshaped = tf.reshape(fc, (-1, 4, 4, 512))
        trans_bn1 = self.bn1_tr(fc_reshaped, training=training)  # 512
        trans_activation1 = self.activation_Relu(trans_bn1)

        # 转置卷积
        trans_conv1 = self.trans_conv1(trans_activation1)    # 256
        trans_bn2 = self.bn2_tr(trans_conv1, training=training)
        trans_activation2 = self.activation_Relu(trans_bn2)

        trans_conv2 = self.trans_conv2(trans_activation2)    # 128
        trans_bn3 = self.bn3_tr(trans_conv2, training=training)
        trans_activation3 = self.activation_Relu(trans_bn3)

        trans_conv3 = self.trans_conv3(trans_activation3)    # 64
        trans_bn4 = self.bn4_tr(trans_conv3, training=training)
        trans_activation4 = self.activation_Relu(trans_bn4)

        trans_conv4 = self.trans_conv4(trans_activation4)    # 3
        output = self.out(trans_conv4)

        return output


class Discriminator(keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        # 卷积层
        self.conv1 = layers.Conv2D(64, 5, strides=2, padding='SAME', activation=None)
        self.conv2 = layers.Conv2D(128, 5, strides=2, padding='SAME', activation=None)
        self.conv3 = layers.Conv2D(256, 5, strides=2, padding='SAME', activation=None)
        self.conv4 = layers.Conv2D(512, 5, strides=2, padding='SAME', activation=None)
        # 批量归一化层
        self.bn1 = layers.BatchNormalization(scale=False)
        self.bn2 = layers.BatchNormalization(scale=False)
        self.bn3 = layers.BatchNormalization(scale=False)
        # Leaky激活层
        self.activation = layers.LeakyReLU(alpha=alpha_Leaky)
        # 展平层
        self.flatten = layers.Flatten()
        self.fc = layers.Dense(units=1, activation=None)
        self.out = layers.Activation(activation='sigmoid')

    def call(self, inputs, training=True, mask=None):
        conv1 = self.conv1(inputs)
        activation1 = self.activation(conv1)

        conv2 = self.conv2(activation1)
        bn1 = self.bn1(conv2)
        activation2 = self.activation(bn1)

        conv3 = self.conv3(activation2)
        bn2 = self.bn2(conv3)
        activation3 = self.activation(bn2)

        conv4 = self.conv4(activation3)
        bn3 = self.bn3(conv4)
        activation4 = self.activation(bn3)

        flat = self.flatten(activation4)
        logits = self.fc(flat)
        out = self.out(logits)

        return out, logits


