import tensorflow as tf
import numpy as np
import os
import random

from collections import namedtuple
from tqdm import tqdm
from glob import glob

from module import batch_norm, instance_norm, conv2d, deconv2d, relu, lrelu, tanh, generator, discriminator, wgan_gp_loss, gan_loss, cls_loss, recon_loss
from util import load_data_list, attr_extract, preprocess_attr, preprocess_image, preprocess_input, save_images
from wav_util import load_FFT_attr


class stargan(object):
    def __init__(self, sess, args):
        #
        self.sess = sess
        self.phase = args.phase  # train or test
        self.data_dir = args.data_dir  # ./data/celebA
        self.log_dir = args.log_dir  # ./assets/log
        self.ckpt_dir = args.ckpt_dir  # ./assets/checkpoint
        self.sample_dir = args.sample_dir  # ./assets/sample
        self.test_dir = args.test_dir  # ./assets/test
        self.epoch = args.epoch  # 100
        self.batch_size = args.batch_size  # 16
        self.image_size = args.image_size  # 64
        self.image_channel = args.image_channel  # 3
        self.nf = args.nf  # 64
        self.n_label = args.n_label  # 10
        self.lambda_gp = args.lambda_gp
        self.lambda_cls = args.lambda_cls  # 1
        self.lambda_rec = args.lambda_rec  # 10
        self.lr = args.lr  # 0.0001
        self.beta1 = args.beta1  # 0.5
        self.continue_train = args.continue_train  # False
        self.snapshot = args.snapshot  # 100
        self.adv_type = args.adv_type  # WGAN or GAN
        self.binary_attrs = args.binary_attrs

        self.attr_keys = ['Male', 'Female', 'KizunaAI', 'Nekomasu', 'Mirai', 'Shiro', 'Kaguya']
#        avaiable attibutes
#        ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips',
#         'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby',
#         'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male',
#         'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose',
#         'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
#         'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']

        # hyper-parameter for building the module
        OPTIONS = namedtuple(
            'OPTIONS', ['batch_size', 'image_size', 'nf', 'n_label', 'lambda_gp'])
        self.options = OPTIONS(
            self.batch_size, self.image_size, self.nf, self.n_label, self.lambda_gp)

        # build model & make checkpoint saver
        self.build_model()
        self.saver = tf.train.Saver()

    def build_model(self):
        # placeholder
        # input_image: A, target_image: B
        self.real_A = tf.placeholder(tf.float32,[None, self.image_size, self.image_size,self.image_channel + self.n_label],name='input_images')
        self.real_B = tf.placeholder(tf.float32,[None, self.image_size, self.image_size,self.image_channel + self.n_label],name='target_images')
        self.attr_B = tf.placeholder(tf.float32, [None, self.n_label], name='target_attr')

        self.fake_B_sample = tf.placeholder(tf.float32,[None, self.image_size,self.image_size
        , self.image_channel],name='fake_images_sample')  # use when updating discriminator

        self.epsilon = tf.placeholder(
            tf.float32, [None, 1, 1, 1], name='gp_random_num')
        self.lr_decay = tf.placeholder(tf.float32, None, name='lr_decay')

        print(np.shape(self.real_A))
        # generate image
        self.fake_B = generator(self.real_A, self.options, False, name='gen')
        print(np.shape(self.fake_B))
        self.fake_A = generator(tf.concat([self.fake_B, self.real_A[:, :, :, self.image_channel:]], axis=3), self.options, True, name='gen')

        # discriminate image
        # src: real or fake, cls: domain classification
        self.src_real_B, self.cls_real_B = discriminator(self.real_B[:, :, :, :self.image_channel],self.options, False, name='disc')
        self.g_src_fake_B, self.g_cls_fake_B = discriminator(self.fake_B, self.options, True, name='disc')  # use when updating generator
        self.d_src_fake_B, self.d_cls_fake_B = discriminator(self.fake_B_sample, self.options, True, name='disc')  # use when updating discriminator

        # loss
        ## discriminator loss ##
        # adversarial loss
        if self.adv_type == 'WGAN':
            gp_loss = wgan_gp_loss(self.real_B[:, :, :, :self.image_channel], self.fake_B_sample, self.options, self.epsilon)
            self.d_adv_loss = tf.reduce_mean(self.d_src_fake_B) - tf.reduce_mean(self.src_real_B) + gp_loss
        else:  # 'GAN'
            d_real_adv_loss = gan_loss(self.src_real_B, tf.ones_like(self.src_real_B))
            d_fake_adv_loss = gan_loss(self.d_src_fake_B, tf.zeros_like(self.d_src_fake_B))
            self.d_adv_loss = d_real_adv_loss + d_fake_adv_loss
        # domain classification loss
        self.d_real_cls_loss = cls_loss(self.cls_real_B, self.attr_B)
        # disc loss function
        self.d_loss = self.d_adv_loss + self.lambda_cls * self.d_real_cls_loss

        ## generator loss ##
        # adv loss
        if self.adv_type == 'WGAN':
            self.g_adv_loss = -tf.reduce_mean(self.g_src_fake_B)
        else:  # 'GAN'
            self.g_adv_loss = gan_loss(
                self.g_src_fake_B, tf.ones_like(self.g_src_fake_B))
        # domain classificatioin loss
        self.g_fake_cls_loss = cls_loss(self.g_cls_fake_B, self.attr_B)
        # reconstruction loss
        self.g_recon_loss = recon_loss(
            self.real_A[:, :, :, :self.image_channel], self.fake_A)
        # gen loss function
        self.g_loss = self.g_adv_loss + self.lambda_cls * \
            self.g_fake_cls_loss + self.lambda_rec * self.g_recon_loss

        # trainable variables
        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'disc' in var.name]
        self.g_vars = [var for var in t_vars if 'gen' in var.name]
#        for var in t_vars: print(var.name)

        # optimizer
        self.d_optim = tf.train.AdamOptimizer(
            self.lr * self.lr_decay, beta1=self.beta1).minimize(self.d_loss, var_list=self.d_vars)
        self.g_optim = tf.train.AdamOptimizer(
            self.lr * self.lr_decay, beta1=self.beta1).minimize(self.g_loss, var_list=self.g_vars)

    def train(self):
        # summary setting
        self.summary()

        # load train data list & load attribute data ここで入力読み込み
        # data_dirとattr_extractの書き換えでおそらく入力変更可
        dataA_files = load_data_list(self.data_dir)
        print(np.shape(dataA_files))
        dataB_files = np.copy(dataA_files)
        self.attr_names = ['Male', 'Female', 'KizunaAI', 'Nekomasu', 'Mirai', 'Shiro', 'Kaguya']
        self.attr_list = load_FFT_attr(self.data_dir)
        print(np.shape(self.attr_list))
        # variable initialize
        self.sess.run(tf.global_variables_initializer())

        # load or not checkpoint
        if self.continue_train and self.checkpoint_load():
            print(" [*] before training, Load SUCCESS ")
        else:
            print(" [!] before training, no need to Load ")

        batch_idxs = len(dataA_files) // self.batch_size  # 182599
        count = 0
        # train
        for epoch in range(self.epoch):
            # get lr_decay
            if epoch < self.epoch / 2:
                lr_decay = 1.0
            else:
                lr_decay = (self.epoch - epoch) / (self.epoch / 2)

            # data shuffle
            np.random.shuffle(dataA_files)
            np.random.shuffle(dataB_files)

            for idx in tqdm(range(batch_idxs)):
                count += 1
                #
                dataA_list = dataA_files[idx * self.batch_size: (idx+1) * self.batch_size]
                dataB_list = dataB_files[idx * self.batch_size: (idx+1) * self.batch_size]
                attrA_list = [self.attr_list[int(os.path.basename(val.split('.')[1]))] for val in dataA_list]
                attrB_list = [self.attr_list[int(os.path.basename(val.split('.')[1]))] for val in dataB_list]

                # get batch images and labels
                attrA, attrB = preprocess_attr(self.attr_names, attrA_list, attrB_list, self.attr_keys)
                imgA, imgB = preprocess_image(dataA_list, dataB_list, self.image_size, phase='train')
                dataA, dataB = preprocess_input(imgA, imgB, attrA, attrB, self.image_size, self.n_label)

                # generate fake_B
                feed = {self.real_A: dataA}
                fake_B = self.sess.run(self.fake_B, feed_dict=feed)

                # update D network for 5 times
                for _ in range(5):
                    epsilon = np.random.rand(self.batch_size, 1, 1, 1)
                    feed = {self.fake_B_sample: fake_B, self.real_B: dataB, self.attr_B: np.array(attrB), self.epsilon: epsilon, self.lr_decay: lr_decay}
                    _, d_loss, d_summary = self.sess.run([self.d_optim, self.d_loss, self.d_sum], feed_dict=feed)

                # updatae G network for 1 time
                feed = {self.real_A: dataA, self.real_B: dataB,
                        self.attr_B: np.array(attrB), self.lr_decay: lr_decay}
                _, g_loss, g_summary = self.sess.run([self.g_optim, self.g_loss, self.g_sum], feed_dict=feed)

                # summary
                self.writer.add_summary(g_summary, count)
                self.writer.add_summary(d_summary, count)

                # save checkpoint and samples
                if count % self.snapshot == 0:
                    print("Iter: %06d, g_loss: %4.4f, d_loss: %4.4f" %
                          (count, g_loss, d_loss))

                    # checkpoint
                    self.checkpoint_save(count)

                    # save samples (from test dataset)
                    self.sample_save(count)

    def test(self):
        # check if attribute available
        # binary_attrsでtagを指定しているので長さは同じに
        if not len(self.binary_attrs) == self.n_label:
            print("binary_attr length is wrong! The length should be {}".format(
                self.n_label))
            return

        # variable initialize
        self.sess.run(tf.global_variables_initializer())

        # load or not checkpoint
        if self.phase == 'test' and self.checkpoint_load():
            print(" [*] before training, Load SUCCESS ")
        else:
            print(" [!] before training, no need to Load ")

        # [5,6] with the seequnce of (realA, realB, fakeB), totally 10 set save
        # data_dirから適当なサンプルを十持ってきている
        # 音声データだから連続的に適当な範囲を持ってくる
        test_files = glob(os.path.join(self.data_dir, 'test', '*'))
        testA_list = random.sample(test_files, 10)

        # get batch images and labels
#        self.attr_keys = ['Black_Hair','Blond_Hair','Brown_Hair', 'Male', 'Young','Mustache','Pale_Skin']
        attrA = [float(i) for i in list(self.binary_attrs)] * len(testA_list)
        imgA, _ = preprocess_image(testA_list, testA_list, self.image_size, phase='test')
        dataA, _ = preprocess_input(imgA, imgA, attrA, attrA, self.image_size, self.n_label)

        # generate fakeB
        # 生成結果はfake_Bの中
        feed = {self.real_A: dataA}
        fake_B = self.sess.run(self.fake_B, feed_dict=feed)

        # save samples
        test_file = os.path.join(self.test_dir, 'test.jpg')
        save_images(imgA, imgA, fake_B, self.image_size, test_file, num=10)

    def summary(self):
        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

        # session : discriminator
        sum_d_1 = tf.summary.scalar('disc/adv_loss', self.d_adv_loss)
        sum_d_2 = tf.summary.scalar('disc/real_cls_loss', self.d_real_cls_loss)
        sum_d_3 = tf.summary.scalar('disc/d_loss', self.d_loss)
        self.d_sum = tf.summary.merge([sum_d_1, sum_d_2, sum_d_3])

        # session : generator
        sum_g_1 = tf.summary.scalar('gen/adv_loss', self.g_adv_loss)
        sum_g_2 = tf.summary.scalar('gen/fake_cls_loss', self.g_fake_cls_loss)
        sum_g_3 = tf.summary.scalar('gen/recon_loss', self.g_recon_loss)
        sum_g_4 = tf.summary.scalar('gen/g_loss', self.g_loss)
        self.g_sum = tf.summary.merge([sum_g_1, sum_g_2, sum_g_3, sum_g_4])

    def checkpoint_load(self):
        print(" [*] Reading checkpoint...")

        ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(self.ckpt_dir, ckpt_name))
            return True
        else:
            return False

    def checkpoint_save(self, step):
        model_name = "stargan.model"
        self.saver.save(self.sess,
                        os.path.join(self.ckpt_dir, model_name),
                        global_step=step)

    def sample_save(self, step):
        test_files = glob(os.path.join(self.data_dir, 'test', '*'))

        # [5,6] with the seequnce of (realA, realB, fakeB), totally 10 set save
        testA_list = random.sample(test_files, 10)
        testB_list = random.sample(test_files, 10)
        attrA_list = [self.attr_list[os.path.basename(val)] for val in testA_list]
        attrB_list = [self.attr_list[os.path.basename(val)] for val in testB_list]

        # get batch images and labels
        attrA, attrB = preprocess_attr(self.attr_names, attrA_list, attrB_list, self.attr_keys)
        imgA, imgB = preprocess_image(testA_list, testB_list, self.image_size, phase='test')
        dataA, _ = preprocess_input(imgA, imgB, attrA, attrB, self.image_size, self.n_label)

        # generate fakeB
        feed = {self.real_A: dataA}
        fake_B = self.sess.run(self.fake_B, feed_dict=feed)

        # save samples
        sample_file = os.path.join(self.sample_dir, '%06d.jpg' % (step))
        save_images(imgA, imgB, fake_B, self.image_size, sample_file, num=10)
