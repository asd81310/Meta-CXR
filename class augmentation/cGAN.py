"""
Conditional GAN model
Bioinformatics, Politecnico di Torino
Authors: Gilberto Manunza, Silvia Giammarinaro
"""
 
import os
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
 
 
class cGAN():
 
        def __init__(self,
                      n_epochs=350,
                      batch_size=128,
                      input_shape=(128, 128, 1),
                      latent_size=100,
                      n_classes = 3,
                      alpha=0.2,
                      drop_rate=0.5,
                      discriminator_lr=8e-5,
                      generator_lr=1e-4,
                      logging_step=10,
                      out_images_path='/content/drive/MyDrive/BIOINF/checkpoints_GAN/cGAN/outImages',
                      checkpoint_dir='/content/drive/MyDrive/BIOINF/checkpoints_GAN/cGAN',
                      use_residual=False):
 
            self.n_epochs = n_epochs
            self.batch_size = batch_size
            self.input_shape = input_shape
            self.latent_size = latent_size
            self.n_classes = n_classes
            self.alpha = alpha
            self.drop_rate = drop_rate
            self.discriminator_lr = discriminator_lr
            self.generator_lr = generator_lr
            self.logging_step = logging_step
            self.out_images_path = out_images_path
            self.checkpoint_dir = checkpoint_dir
            self.use_residual = use_residual


            self.model = self._build_model()
 
        def create_discriminator(self):                
            leaky = tf.keras.layers.LeakyReLU(self.alpha)

            input_image = Input(self.input_shape)

            input_label = Input(shape=(self.n_classes,))

            # Embedding for categorical input
            #li = Embedding(self.n_classes, 50)(input_label)

            # Scale embedding to image size
            n_nodes = self.input_shape[0] * self.input_shape[1]
            li = Dense(50)(input_label)
            li = Dense(n_nodes)(li)
            
            # Reshape to image input size
            li = Reshape((self.input_shape[0], self.input_shape[1], 1))(li)
            
            # Concatenate input image and label
            merge = Concatenate()([input_image, li])

            conv1 = Conv2D(32, 3, activation=leaky, padding='same', kernel_initializer='he_normal')(merge)
            conv1 = Dropout(0.2)(conv1)

            conv1 = Conv2D(32, 3, activation=leaky, padding='same', kernel_initializer='he_normal')(conv1)
            pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

            conv2 = Conv2D(64, 3, activation=leaky, padding='same', kernel_initializer='he_normal')(pool1)
            conv2 = Dropout(0.2)(conv2)

            conv2 = Conv2D(64, 3, activation=leaky, padding='same', kernel_initializer='he_normal')(conv2)
            pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

            conv3 = Conv2D(128, 3, activation=leaky, padding='same', kernel_initializer='he_normal')(pool2)
            conv3 = Dropout(0.2)(conv3)

            conv3 = Conv2D(128, 3, activation=leaky, padding='same', kernel_initializer='he_normal')(conv3)
            pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

            conv4 = Conv2D(256, 3, activation=leaky, padding='same', kernel_initializer='he_normal')(pool3)
            conv4 = Dropout(0.2)(conv4)

            conv4 = Conv2D(256, 3, activation=leaky, padding='same', kernel_initializer='he_normal')(conv4)
            pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

            conv5 = Conv2D(512, 3, activation=leaky, padding='same', kernel_initializer='he_normal')(pool4)
            conv5 = Dropout(0.2)(conv5)

            conv5 = Conv2D(512, 3, activation=leaky, padding='same', kernel_initializer='he_normal')(conv5)
            drop5 = Dropout(0.5)(conv5)

            gap1 = GlobalAveragePooling2D()(drop5)

            fc1 = Dense(128)(gap1)
            outputs = Dense(1)(fc1)

            model = Model(inputs=[input_image, input_label], outputs=outputs)
            return model
 
        def create_generator(self):
 
            width = self.input_shape[0] #128
            height = self.input_shape[1] #128
            channels = self.input_shape[2] #1
            leaky = tf.keras.layers.LeakyReLU(self.alpha)


            dim1 = width // 16 #8
            dim2 = height // 16 #8

            in_label = Input(shape=(1,))
            # embedding for categorical input
            li = Embedding(self.n_classes, 10)(in_label)
            # embedding con 3/1 -> comportamento simile a non condizionale

            n_nodes = dim1 * dim2
            li = Dense(n_nodes)(li)
            li = Reshape((dim1, dim2, 1))(li)

            in_lat = Input(shape=(self.latent_size,))
            x = Dense( dim1 * dim2 * 12, activation="relu")(in_lat)
            x = BatchNormalization()(x)

            x = Reshape((dim1, dim2, 12))(x)

            merge = Concatenate()([x, li])

            # now add conv 2D transpose: transposed convoultional or deconvolution
            #x_shortcut = merge
            x = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding="same", activation=leaky)(merge)
            x = BatchNormalization()(x)
            x = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same")(x)
            x = BatchNormalization()(x)
            #x = Add()([merge, x_shortcut])

            x = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding="same", activation=leaky)(x)
            x = BatchNormalization()(x)
            x = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding="same")(x)
            x = BatchNormalization()(x)

            # now add final layer
            outputs = Conv2DTranspose(channels, (3, 3), strides=(1, 1), padding="same", activation="tanh")(x)

            model = Model(inputs=[in_lat, in_label], outputs=outputs)
            return model
 
        def create_residual_generator(self):
            leaky = tf.keras.layers.LeakyReLU(self.alpha)

            input_noise = Input(shape=self.latent_size)

            input_label = Input(shape=(self.n_classes,))

            # Embedding for categorical input
            #li = Embedding(self.n_classes, 50)(input_label)
            li = Dense(50)(input_label)
            # Match initial image size
            n_nodes = 8 * 8
            li = Dense(n_nodes)(li)
            # reshape to add additional channel
            li = Reshape((8, 8, 1))(li)

            dense1 = Dense(8*8*256, use_bias=False, input_shape=(self.latent_size,))(input_noise)
            dense1 = BatchNormalization()(dense1)
            dense1 = leaky(dense1)
            dense1 = Reshape((8, 8, 256))(dense1)

            merge = Concatenate()([dense1, li])

            up1 = Conv2DTranspose(256, 2, strides=(2, 2), activation=leaky, padding='same', kernel_initializer='he_normal')(merge)
            conv1 = Conv2D(256, 3, activation=leaky, padding='same', kernel_initializer='he_normal')(up1)
            conv1 = Dropout(0.2)(conv1)
            conv1 = Conv2D(512, 3, activation=leaky, padding='same', kernel_initializer='he_normal')(conv1)

            # Residual connection
            up_res_1 = UpSampling2D(size=(2,2))(up1)

            up2 = Conv2DTranspose(128, 2, strides=(2, 2), activation=leaky, padding='same', kernel_initializer='he_normal')(conv1)
            merge2 = concatenate([up_res_1,up2], axis = 3)
            conv2 = Conv2D(128, 3, activation=leaky, padding='same', kernel_initializer='he_normal')(merge2)
            conv2 = Dropout(0.2)(conv2)
            conv2 = Conv2D(128, 3, activation=leaky, padding='same', kernel_initializer='he_normal')(conv2)

            # Residual connection
            up_res_2 = UpSampling2D(size=(2,2))(up2)


            up3 = Conv2DTranspose(64, 2, strides=(2, 2), activation=leaky, padding='same', kernel_initializer='he_normal')(conv2)
            merge3 = concatenate([up_res_2,up3], axis = 3)
            conv3= Conv2D(64, 3, activation=leaky, padding='same', kernel_initializer='he_normal')(merge3)
            conv3= Dropout(0.2)(conv3)
            conv3= Conv2D(64, 3, activation=leaky, padding='same', kernel_initializer='he_normal')(conv3)

            # Residual connection
            up_res_3 = UpSampling2D(size=(2,2))(up3)

            up4 = Conv2DTranspose(32, 2, strides=(2, 2), activation=leaky, padding='same', kernel_initializer='he_normal')(conv3)
            merge4 = concatenate([up_res_3,up4], axis = 3)
            conv4 = Conv2D(32, 3, activation=leaky, padding='same', kernel_initializer='he_normal')(merge4)
            conv4 = Dropout(0.2)(conv4)
            conv4 = Conv2D(32, 3, activation=leaky, padding='same', kernel_initializer='he_normal')(conv4)
            conv4 = Dropout(0.2)(conv4)
            conv4 = Conv2D(2, 3, activation=leaky, padding='same', kernel_initializer='he_normal')(conv4)

            output = Conv2D(1, (1, 1), activation='tanh', padding='same', name="out_dec")(conv4)
            model = Model(inputs=[input_noise, input_label], outputs=[output])
            return model
 
        class _cGANModel(keras.Model):
            def __init__(self, discriminator, generator, latent_size, num_classes):
                    super(cGAN._cGANModel, self).__init__()
                    self.discriminator = discriminator
                    self.generator = generator
                    self.latent_size = latent_size
                    self.num_classes = num_classes

                    self.loss_tracker_generator = keras.metrics.Mean(name="gen_loss")
                    self.loss_tracker_discriminator = keras.metrics.Mean(name="disc_loss")
                    self.loss_true_tracker_discriminator = keras.metrics.Mean(name="disc_loss_real")
                    self.loss_fake_tracker_discriminator = keras.metrics.Mean(name="disc_loss_fake")
                    self.accuracy_real_tracker_discriminator = keras.metrics.Mean(name="disc_acc_real")
                    self.accuracy_fake_tracker_discriminator = keras.metrics.Mean(name="disc_acc_fake")
    

            def call(self, x):
                return x

            def compile(self, discriminator_optimizer, generator_optimizer):
                super(cGAN._cGANModel, self).compile()
                self.generator_optimizer = generator_optimizer
                self.discriminator_optimizer = discriminator_optimizer

            # Define and element-wise binary cross entropy loss
            def element_wise_cross_entropy_from_logits(self, labels, logits):
                # Compute the loss element-wise
                losses = tf.nn.sigmoid_cross_entropy_with_logits(labels = labels, logits = logits)
                # Compute average to reduce everything to a specific number
                loss = tf.reduce_mean(losses)
                return loss

            def generator_loss(self, fake_output):
                return self.element_wise_cross_entropy_from_logits(tf.ones_like(fake_output), fake_output)

            def discriminator_loss(self, real_output, fake_output, images):
                # label smoothing added to real_loss
                real_loss = self.element_wise_cross_entropy_from_logits(tf.ones_like(real_output), real_output)
                fake_loss = self.element_wise_cross_entropy_from_logits(tf.zeros_like(fake_output), fake_output)

                total_loss = real_loss + fake_loss
                return real_loss, fake_loss, total_loss
        

            def train_step(self, data):
                images, labels = data
                noise = tf.random.normal([images.shape[0], self.latent_size])
                fake_labels = np.random.randint(0, self.num_classes, labels.shape[0])
                fake_labels = tf.one_hot(fake_labels, depth=self.num_classes)
                with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

                        generated_images = self.generator((noise, fake_labels), training=True)

                        real_output = self.discriminator((images, labels), training=True)
                        fake_output = self.discriminator((generated_images, fake_labels), training=True)

                        gen_loss = self.generator_loss(fake_output)#ok
                        
                        disc_real_loss, disc_fake_loss, disc_loss = self.discriminator_loss(real_output, fake_output, images)

                gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
                gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

                self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
                self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

                # Compute metrics
                self.loss_tracker_generator.update_state(gen_loss)
                self.loss_tracker_discriminator.update_state(disc_loss)
                self.loss_true_tracker_discriminator.update_state(disc_real_loss)
                self.loss_fake_tracker_discriminator.update_state(disc_fake_loss)

                
                preds_real = tf.round(tf.sigmoid(real_output))
                accuracy_real = tf.math.reduce_mean(tf.cast(tf.math.equal(preds_real, tf.ones_like(preds_real)), tf.float32))
                self.accuracy_real_tracker_discriminator.update_state(accuracy_real)

                preds_fake = tf.round(tf.sigmoid(fake_output))
                accuracy_fake = tf.math.reduce_mean(tf.cast(tf.math.equal(preds_fake, tf.zeros_like(preds_fake)), tf.float32))
                self.accuracy_fake_tracker_discriminator.update_state(accuracy_fake)

                return {'gen_loss': self.loss_tracker_generator.result(), 'disc_loss': self.loss_tracker_discriminator.result(), \
                        'disc_loss_real': self.loss_true_tracker_discriminator.result(), 'disc_loss_fake': self.loss_fake_tracker_discriminator.result(), \
                        'disc_acc_real': self.accuracy_real_tracker_discriminator.result(), 'disc_acc_fake': self.accuracy_fake_tracker_discriminator.result()}

            def test_step(self, data):
                pass

            @property
            def metrics(self):
                # We list our `Metric` objects here so that `reset_states()` can be
                # called automatically at the start of each epoch
                # or at the start of `evaluate()`.
                # If you don't implement this property, you have to call
                # `reset_states()` yourself at the time of your choosing.
                return [self.loss_tracker_generator, self.loss_tracker_discriminator, \
                        self.loss_true_tracker_discriminator, self.loss_fake_tracker_discriminator, \
                        self.accuracy_real_tracker_discriminator, self.accuracy_fake_tracker_discriminator]

        def _build_model(self):
            if self.use_residual:
                self.generator = self.create_residual_generator()
            else:
                 self.generator = self.create_generator()

            self.discriminator = self.create_discriminator()

            model = self._cGANModel(generator=self.generator, discriminator=self.discriminator, latent_size=self.latent_size, num_classes=self.n_classes)

            self.generator_optimizer = tf.keras.optimizers.Adam(self.generator_lr, beta_1=0.5, clipvalue=5)
            self.discriminator_optimizer = tf.keras.optimizers.Adam(self.discriminator_lr, beta_1=0.5)

            model.compile(generator_optimizer=self.generator_optimizer, discriminator_optimizer=self.discriminator_optimizer)

            return model

        def generate_latent_points(self):
            # generate points in the latent space
            x_input = np.random.randn(self.latent_size * self.batch_size)
            # reshape into a batch of inputs for the network
            x_input = x_input.reshape(self.batch_size, self.latent_size )
            # generate labels
            labels = np.random.randint(0, self.n_classes, self.batch_size)
            return [x_input, labels]   
            
        def train_model(self, train_ds, benchmark_noise, benchmark_labels):
            # set checkpoint directory
            checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
            checkpoint = tf.train.Checkpoint(generator_optimizer=self.model.generator_optimizer,
                                                discriminator_optimizer=self.model.discriminator_optimizer,
                                                model=self.model)
        
            # creating dictionaries for history and accuracy for the plots
            self.history = {}
            self.history['G loss'] = []
            self.history['D loss'] = []
            self.history['D loss Real'] = []
            self.history['D loss Fake'] = []
            self.accuracy = {}
            self.accuracy['D accuracy Real'] = []
            self.accuracy['D accuracy Fake'] = []

            print("Starting training of the cGAN model.")

            print("Batches per epoch ", len(train_ds))

            for epoch in range(self.n_epochs+1):
                # Keep track of the losses at each step
                epoch_gen_loss = []
                epoch_disc_loss = []
                epoch_disc_loss_true = []
                epoch_disc_loss_fake = []
                epoch_disc_acc_true = []
                epoch_disc_acc_fake = []

                print(f"Starting epoch {epoch} of {self.n_epochs}")

                for step, batch in enumerate(train_ds):
          
                    images, labels = batch
                    #labels = tf.math.argmax(labels,1)
                    gen_loss_step, disc_loss_step, disc_loss_true_step, disc_loss_fake_step, disc_acc_true_step, disc_acc_fake_step = self.model.train_on_batch(images, labels)

                    epoch_gen_loss.append(gen_loss_step)
                    epoch_disc_loss.append(disc_loss_step)
                    epoch_disc_loss_true.append(disc_loss_true_step)
                    epoch_disc_loss_fake.append(disc_loss_fake_step)
                    epoch_disc_acc_true.append(disc_acc_true_step)
                    epoch_disc_acc_fake.append(disc_acc_fake_step)

                    if step % self.logging_step == 0:
                        print(f"\tLosses at step {step}:")
                        print(f"\t\tGenerator Loss: {gen_loss_step}")
                        print(f"\t\tDiscriminator Loss: {disc_loss_step}")
                        print(f"\t\tAccuracy Real: {disc_acc_true_step}")
                        print(f"\t\tAccuracy Fake: {disc_acc_fake_step}")


                if epoch % self.logging_step == 0:
                    generator_images = self.model.generator((benchmark_noise, benchmark_labels), training=False)

                    print("Generated images: ")
                    self.plot_fake_figures(generator_images, benchmark_labels, 4, epoch,  self.out_images_path)

                if (epoch % (self.logging_step*5)) == 0:
                    checkpoint.save(file_prefix = checkpoint_prefix)
                    self.generator.save('mimic_gen/model' +str(epoch))

                self.history['G loss'].append(np.array(epoch_gen_loss).mean())
                self.history['D loss'].append(np.array(epoch_disc_loss).mean())
                self.history['D loss Real'].append(np.array(epoch_disc_loss_true).mean())          
                self.history['D loss Fake'].append(np.array(epoch_disc_loss_fake).mean())     
                self.accuracy['D accuracy Real'].append(np.array(epoch_disc_acc_true).mean())     
                self.accuracy['D accuracy Fake'].append(np.array(epoch_disc_acc_fake).mean())

        def plot_losses(self, data, xaxis, yaxis, ylim=0):
            pd.DataFrame(data).plot(figsize=(10,8))
            plt.grid(True)
            plt.xlabel(xaxis)
            plt.ylabel(yaxis)
            if ylim!=0:
                plt.ylim(0, ylim)
            plt.show()
 
        @staticmethod
        def plot_fake_figures(x, labels, n, epoch, dir='./'):
 
            labels_dict = {
                    0: "Cardiomegaly",
                    1: "Consolidation",
                    2: "Edema",
                    3: "Pleural Effusion",
                    4: "No Finding"
            }

            fig = plt.figure(figsize=(6,6))
            labels = np.argmax(labels,1)
            for i in range(n*n):
                plt.subplot(n,n,i+1)
                plt.xticks([])
                plt.yticks([])
                img=x[i,:,:,:]
                # rescale for visualization purposes
                img = tf.keras.preprocessing.image.array_to_img(img)
                plt.imshow(img, cmap="gray")
                plt.xlabel(labels_dict[labels[i]])
                plt.imshow(img, cmap='gray')
                plt.savefig('{}/image_at_epoch_{:04d}.png'.format(dir, epoch))

            plt.show()