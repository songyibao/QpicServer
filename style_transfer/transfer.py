import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from style_transfer.model import NeuralStyleTransferModel
import style_transfer.settings as settings
import style_transfer.utils as utils

class NeuralStyleTransfer:
    def __init__(self,contentImagepath,styleImagePath):
        self.model = NeuralStyleTransferModel()
        self.content_image = utils.load_images(contentImagepath)
        self.style_image = utils.load_images(styleImagePath)
        self.target_content_features = self.model([self.content_image, ])['content']
        self.target_style_features = self.model([self.style_image, ])['style']

        self.M = settings.WIDTH * settings.HEIGHT
        self.N = 3

        self.optimizer = tf.keras.optimizers.Adam(settings.LEARNING_RATE)
        self.noise_image = tf.Variable((self.content_image + np.random.uniform(-0.2, 0.2, (1, settings.HEIGHT, settings.WIDTH, 3))) / 2)

    def gram_matrix(self, feature):
        x = tf.transpose(feature, perm=[2, 0, 1])
        x = tf.reshape(x, (x.shape[0], -1))
        return x @ tf.transpose(x)

    def _compute_content_loss(self, noise_features, target_features):
        content_loss = tf.reduce_sum(tf.square(noise_features - target_features))
        x = 2. * self.M * self.N
        content_loss = content_loss / x
        return content_loss

    def compute_content_loss(self, noise_content_features):
        content_losses = []
        for (noise_feature, factor), (target_feature, _) in zip(noise_content_features, self.target_content_features):
            layer_content_loss = self._compute_content_loss(noise_feature, target_feature)
            content_losses.append(layer_content_loss * factor)
        return tf.reduce_sum(content_losses)

    def _compute_style_loss(self, noise_feature, target_feature):
        noise_gram_matrix = self.gram_matrix(noise_feature)
        style_gram_matrix = self.gram_matrix(target_feature)
        style_loss = tf.reduce_sum(tf.square(noise_gram_matrix - style_gram_matrix))
        x = 4. * (self.M ** 2) * (self.N ** 2)
        return style_loss / x

    def compute_style_loss(self, noise_style_features):
        style_losses = []
        for (noise_feature, factor), (target_feature, _) in zip(noise_style_features, self.target_style_features):
            layer_style_loss = self._compute_style_loss(noise_feature, target_feature)
            style_losses.append(layer_style_loss * factor)
        return tf.reduce_sum(style_losses)

    def total_loss(self, noise_features):
        content_loss = self.compute_content_loss(noise_features['content'])
        style_loss = self.compute_style_loss(noise_features['style'])
        return content_loss * settings.CONTENT_LOSS_FACTOR + style_loss * settings.STYLE_LOSS_FACTOR

    @tf.function
    def train_one_step(self):
        with tf.GradientTape() as tape:
            noise_outputs = self.model(self.noise_image)
            loss = self.total_loss(noise_outputs)
        grad = tape.gradient(loss, self.noise_image)
        self.optimizer.apply_gradients([(grad, self.noise_image)])
        return loss

    def start(self, filename_only):
        if not os.path.exists(settings.OUTPUT_DIR):
            os.mkdir(settings.OUTPUT_DIR)

        for epoch in range(settings.EPOCHS):
            with tqdm(total=settings.STEPS_PER_EPOCH, desc='Epoch {}/{}'.format(epoch + 1, settings.EPOCHS)) as pbar:
                for step in range(settings.STEPS_PER_EPOCH):
                    _loss = self.train_one_step()
                    pbar.set_postfix({'loss': '%.4f' % float(_loss)})
                    pbar.update(1)

        utils.save_image(self.noise_image, '{}/{}.jpg'.format(settings.OUTPUT_DIR, filename_only))
