import tensorflow as tf
import numpy as np
import glob, random, os
from PIL import Image
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

model_path = "saved_models/"
model_name = model_path + 'model'

class Network(object):
    # Create model
    def __init__(self):
        self.image = tf.placeholder(tf.float32, [None, 100, 240, 3], name='image')
        self.target = tf.placeholder(tf.float32, [None, 100, 240, 3], name='image')

        self.resized_target = tf.image.resize_images(self.target, [64, 64])
        self.resized_image = tf.image.resize_images(self.image, [64, 64])
        tf.summary.image('resized_image', self.resized_image, 5)

        self.z_mu, self.z_logvar = self.encoder(self.resized_image)
        self.z = self.sample_z(self.z_mu, self.z_logvar)
        self.reconstructions = self.decoder(self.z)
        tf.summary.image('targets', self.resized_target, 5)
        tf.summary.image('reconstructions', self.reconstructions, 5)

        self.merged = tf.summary.merge_all()

        self.loss = self.compute_loss()

    def sample_z(self, mu, logvar):
        eps = tf.random_normal(shape=tf.shape(mu))
        return mu + tf.exp(logvar / 2) * eps

    def encoder(self, x):
        x = tf.layers.conv2d(x, filters=32, kernel_size=4, strides=2, padding='valid', activation=tf.nn.relu)
        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding='valid', activation=tf.nn.relu)
        x = tf.layers.conv2d(x, filters=128, kernel_size=4, strides=2, padding='valid', activation=tf.nn.relu)
        x = tf.layers.conv2d(x, filters=256, kernel_size=4, strides=2, padding='valid', activation=tf.nn.relu)

        x = tf.layers.flatten(x)
        z_mu = tf.layers.dense(x, units=32, name='z_mu')
        z_logvar = tf.layers.dense(x, units=32, name='z_logvar')
        return z_mu, z_logvar

    def decoder(self, z):
        x = tf.layers.dense(z, 1024, activation=None)
        x = tf.reshape(x, [-1, 1, 1, 1024])
        x = tf.layers.conv2d_transpose(x, filters=128, kernel_size=5, strides=2, padding='valid', activation=tf.nn.relu)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=5, strides=2, padding='valid', activation=tf.nn.relu)
        x = tf.layers.conv2d_transpose(x, filters=32, kernel_size=6, strides=2, padding='valid', activation=tf.nn.relu)
        x = tf.layers.conv2d_transpose(x, filters=3, kernel_size=6, strides=2, padding='valid', activation=tf.nn.sigmoid)
        return x

    def compute_loss(self):
        logits_flat = tf.layers.flatten(self.reconstructions)
        labels_flat = tf.layers.flatten(self.resized_target)
        reconstruction_loss = tf.reduce_sum(tf.square(logits_flat - labels_flat), axis = 1)
        kl_loss = 0.5 * tf.reduce_sum(tf.exp(self.z_logvar) + self.z_mu**2 - 1. - self.z_logvar, 1)
        vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
        return vae_loss


def load_image(path):
    img = Image.open(path) #.convert('L')
    arr = np.array(img, np.float32) / 255
    if (len(arr) == 120):
        raise Error()
    return arr


def data_iterator(batch_size):
    # HERE LOAD IMAGES AS NP ARRAYS
    # ORIGINALS 96x96

    # TUB[]
    data_files = glob.glob('./green/*_.jpg')
    data = np.array(data_files)
    np.random.shuffle(data)
    np.random.shuffle(data)
    N = data.shape[0]

    while True:
        # img[]
        start = np.random.randint(0, N-batch_size)
        batch_files = data[start:start+batch_size]
        images = []
        targets = []
        for image_file in batch_files:
            try:
                img = load_image(image_file)
                target = load_image('./green/targets/%s' % image_file.split('/')[2])
                images.append(img)
                targets.append(target)
            except:
                continue
        yield images, targets

def train_vae():
    sess = tf.InteractiveSession()

    global_step = tf.Variable(0, name='global_step', trainable=False)

    writer = tf.summary.FileWriter('logdir')

    network = Network()
    train_op = tf.train.AdamOptimizer(0.0001).minimize(network.loss, global_step=global_step)
    tf.global_variables_initializer().run()

    saver = tf.train.Saver(max_to_keep=1)
    step = global_step.eval()
    training_data = data_iterator(batch_size=6)
    try:
        saver.restore(sess, tf.train.latest_checkpoint(model_path))
        print("Model restored from: {}".format(model_path))
    except:
        print("Could not restore saved model")

    try:
        while True:
            images, targets = next(training_data)
            _, loss_value, summary = sess.run([train_op, network.loss, network.merged],
                                feed_dict={network.image: images, network.target: targets})
            writer.add_summary(summary, step)

            if np.isnan(loss_value):
                print('Loss value is NaN')
            if step % 10 == 0 and step > 0:
                print ('step {}: training loss {:.6f}'.format(step, loss_value))
                test_img = load_image("./images/0_cam-image_array_.jpg8.jpg")

                #print(generated)
                cv2.imshow('original', test_img)
                reconstruction = sess.run(network.reconstructions, feed_dict={network.image: [test_img], network.target: [test_img]})
                cv2.imshow('autodecoded', reconstruction[0])

                # Draw overlay
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            if step % 300 == 0 and step > 0:
                save_path = saver.save(sess, model_name, global_step=global_step)
            if loss_value <= 5: # 35:
                print ('step {}: training loss {:.6f}'.format(step, loss_value))
                save_path = saver.save(sess, model_name, global_step=global_step)
                break
            step+=1

    except (KeyboardInterrupt, SystemExit):
        print("Manual Interrupt")

    except Exception as e:
        print("Exception: {}".format(e))

def load_vae():

    graph = tf.Graph()
    with graph.as_default():
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config, graph=graph)

        network = Network()
        init = tf.global_variables_initializer()
        sess.run(init)

        saver = tf.train.Saver(max_to_keep=1)
        training_data = data_iterator(batch_size=128)

        try:
            saver.restore(sess, tf.train.latest_checkpoint(model_path))
        except:
            raise ImportError("Could not restore saved model")

        return sess, network

if __name__ == '__main__':
    train_vae()
