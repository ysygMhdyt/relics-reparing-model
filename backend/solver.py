import os
import tensorflow as tf
import numpy as np
from glob import glob
from matplotlib import pyplot as plt

from model import Generator, Discriminator
from utils import generate_mask
from utils import get_image


class Solver(object):

    def __init__(self, config):
        # 从config中获取所需信息
        self.iters = config.num_iters
        # 裁剪图像的大小
        self.image_size = 64
        # 裁剪图像的形状 全部图像保存为64*64*3的矩阵 再由batch_size，形成四维矩阵
        self.image_shape = [64, 64, 3]
        self.batch_size = config.batch_size

        # 根据使用情景判断，当使用测试情形时，则batch_size = 1
        if config.use_mode == 'train':
            self.data = glob(os.path.join(config.data_dir, '*.jpg'))
        if config.use_mode == 'test':
            self.data = glob(os.path.join(config.test_dir, '*.jpg'))
            # test模式下确保batc_size = 1
            self.batch_size = 1
        # 共有的batch数
        self.batch_num = int(len(self.data) / config.batch_size)
        # 记录当前损失及保存模型的步数
        self.log_step = config.log_step
        self.model_save_step = config.model_save_step
        # 学习率
        self.learning_rate = config.learning_rate
        self.beta1 = config.beta1
        # 各种目录
        self.data_dir = config.data_dir
        self.test_dir = config.test_dir
        self.result_dir = config.result_dir
        self.model_save_dir = config.model_save_dir
        # 用于添加模糊块儿的矩阵
        mask, imask = generate_mask(self.batch_size)
        self.mask = mask
        self.imask = imask

        # 生成器、判别器及其优化器
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.global_counter = tf.compat.v1.train.get_or_create_global_step()
        self.generator_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta1)
        self.discriminator_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                                        beta1=self.beta1)
        # 保存点
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                              discriminator_optimizer=self.discriminator_optimizer,
                                              generator=self.generator,
                                              discriminator=self.discriminator)

    def train(self):
        training_continue = False  # 是否继续训练
        # 直接开始训练
        for i in range(self.iters):
            np.random.shuffle(self.data)
            for ii in range(self.batch_num):
                print("第" + str(i) + "/" + str(self.iters) + "周期,第" + str(ii) + "batch")
                batch, real_part, fake_input = self.get_data_from_class(ii)
                # 梯度更新
                with tf.GradientTape(persistent=True) as tape:
                    # 运行生成器
                    g_result = self.generator(fake_input)
                    # 真图像判别器
                    d_result_real, d_logits_real = self.discriminator(batch)
                    # 假图像判别器
                    d_result_fake, d_logits_fake = self.discriminator(g_result)
                    # 原始生成器损失函数
                    g_loss_t = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                                labels=tf.ones_like(d_result_fake)))
                    # 生成器损失函数上下文部分
                    g_loss_contextual = tf.reduce_mean(tf.abs(g_result - batch))
                    # 最终生成器损失函数
                    g_loss = g_loss_contextual + 0.1 * g_loss_t
                    print(g_loss)
                    # 计算判别器损失函数
                    d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,
                                                                                    labels=tf.ones_like(
                                                                                        d_logits_real) * 0.9)) + \
                             tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                                                    labels=tf.zeros_like(
                                                                                        d_logits_fake)))
                    print(d_loss)
                    if i % self.log_step == 0 and ii == self.batch_num - 1:
                        print("第" + str(i + 1) + "/" + str(self.iters) + "周期,生成器损失:" + str(g_loss) + ",判别器损失:" + str(d_loss))
                    generator_grad = tape.gradient(g_loss,self.generator.variables)
                    discriminator_grad = tape.gradient(d_loss, self.discriminator.variables)
                    self.generator_optimizer.apply_gradients(zip(generator_grad, self.generator.variables),global_step=self.global_counter)
                    self.discriminator_optimizer.apply_gradients(zip(discriminator_grad, self.discriminator.variables), global_step=self.global_counter)

            # 保存模型
            if (i + 1) % self.model_save_step == 0:
                # 保存检查点
                self.checkpoint.save(file_prefix=self.model_save_dir)
                # 获取测试数据
                testdata = glob(os.path.join(self.test_dir, '*.jpg'))
                num = int(len(testdata))
                print("num：",num)
                for i in range(num):
                    print("第"+str(i)+"张图片")
                    testbatch, testrealpart, testfakeinput = self.get_data_from_other(i,testdata,1)

                    with tf.GradientTape(persistent=True) as tape:
                        g_result = self.generator(testfakeinput, training=False)
                        f_part = tf.multiply(g_result, self.imask)  # 取出生成图片的中间部分
                        r_part = tf.multiply(testbatch, self.mask)  # 取出原图像边界部分
                        result = tf.add(f_part, r_part)
                    self.show_and_save_image(g_result[0].numpy(), i, 1)  # 生成器生成的完整图像
                    self.show_and_save_image(testbatch[0], i, 2)  # 原图像
                    self.show_and_save_image(testrealpart[0], i, 3)  # 真实图像
                    self.show_and_save_image(testfakeinput[0], i, 4)  # 输入噪音后的图像
                    self.show_and_save_image(result[0], i, 5)  # 最终拼接后结果


    def test(self):
        self.checkpoint.restore(tf.train.latest_checkpoint(self.model_save_dir))
        np.random.shuffle(self.data)
        for i in range(self.batch_num):
            print("第"+str(i)+"/"+str(self.batch_num)+"张图片")
            batch, real_part, fake_input = self.get_data_from_class(i)

            with tf.GradientTape(persistent=True) as tape:
                g_result = self.generator(fake_input, training=False)
                f_part = tf.multiply(g_result, self.imask)     # 取出生成图片的中间部分
                r_part = tf.multiply(batch, self.mask)         # 取出原图像边界部分
                result = tf.add(f_part, r_part)
            self.show_and_save_image(g_result[0].numpy(), i, 1) # 生成器生成的完整图像
            self.show_and_save_image(batch[0], i, 2)            # 原图像
            self.show_and_save_image(real_part[0], i, 3)        # 真实图像
            self.show_and_save_image(fake_input[0], i, 4)       # 输入噪音后的图像
            self.show_and_save_image(result[0], i, 5)           # 最终拼接后结果

    # 获取第ii次迭代所要用的数据
    def get_data_from_class(self, ii):
        # 随机生成噪音
        fake_input = tf.random.uniform(shape=([self.batch_size] + self.image_shape),
                                       minval=-1.0, maxval=1.0, dtype=tf.float32)
        # 获取中心32*32的噪音块
        fake_part = tf.multiply(fake_input, self.imask)
        batch_files = self.data[ii * self.batch_size:(ii + 1) * self.batch_size]
        batch = []
        for batch_file in batch_files:
            img = get_image(batch_file, self.image_size, is_crop=True)
            img = np.array(img).astype(np.float32).reshape(self.image_shape)    # 转为64*64*3
            batch.append(img)
        batch = np.array(batch)  # shape: batch_file * 64 * 64 * 3
        real_part = tf.multiply(batch, self.mask)  # 获取外围真实部分
        fake_input = tf.add(fake_part, real_part)  # 输入生成器的缺损图片
        return batch, real_part, fake_input

    def get_data_from_other(self, ii, data, batch_size):
        # 随机生成噪音
        fake_input = tf.random.uniform(shape=([batch_size] + self.image_shape),
                                       minval=-1.0, maxval=1.0, dtype=tf.float32)
        # 获取中心32*32的噪音块
        fake_part = tf.multiply(fake_input, self.imask)
        batch_files = data[ii * batch_size:(ii + 1) * batch_size]
        batch = []
        for batch_file in batch_files:
            img = get_image(batch_file, self.image_size, is_crop=True)
            img = np.array(img).astype(np.float32).reshape(self.image_shape)    # 转为64*64*3
            batch.append(img)
        batch = np.array(batch)  # shape: batch_file * 64 * 64 * 3
        real_part = tf.multiply(batch, self.mask)  # 获取外围真实部分
        fake_input = tf.add(fake_part, real_part)  # 输入生成器的缺损图片
        return batch, real_part, fake_input

    def show_and_save_image(self, theImage, namepart1, namepart2):
        plt.xticks([])
        plt.yticks([])
        plt.imshow(theImage)
        plt.savefig(self.result_dir + str(namepart1) + '-' + str(namepart2) + '.jpg')

    def repair(self, img_path):
        # 前置条件 img应已经过预处理
        img = get_image(img_path,image_size=64,is_crop=True)

        fake_input = tf.random.uniform(shape=([self.batch_size] + self.image_shape),
                                       minval=-1.0, maxval=1.0, dtype=tf.float32)
        fake_part = tf.multiply(fake_input, self.imask)
        batch = []
        img = np.array(img).astype(np.float32).reshape(self.image_shape)    # 转为64*64*3
        batch.append(img)
        batch = np.array(batch) # 1*64*64*3
        real_part = tf.multiply(batch,self.mask)
        fake_input = tf.add(fake_part, real_part)

        self.checkpoint.restore(tf.train.latest_checkpoint(self.model_save_dir))  # 获取检查点
        with tf.GradientTape(persistent=True) as tape:
            g_result = self.generator(fake_input, training=False)
            f_part = tf.multiply(g_result, self.imask)  # 取出生成图片的中间部分
            r_part = tf.multiply(batch, self.mask)  # 取出原图像边界部分
            result = tf.add(f_part, r_part)

        all_path = os.path.split(img_path)
        path_front = all_path[0]
        file_name = all_path[1]
        real_file_name = file_name.split('.')[0]+"_repaired.jpg"
        result_path = path_front+"/"+real_file_name
        plt.imshow(result[0].numpy())
        plt.savefig(result_path)

        return result_path

