import tensorflow as tf
import cv2
import numpy as np
import model
import input_file_for_flower as input_file
from  PIL import Image
import matplotlib.pyplot as plt



log_dir = 'D:/csdn/logs/'
#IMG_DIR = r'D:\csdn\flower_photos\dandelion.0 (253).jpg'
#IMG_DIR = r'D:\csdn\flower_photos\daisy.0 (23).jpg'
BATCH_SIZE = 1
N_CLASSES = 5
train_dir = "D:\\csdn\\flower_photos\\"
IMG_W = 28  # resize the image, if the input image is too large, training will be very slow.
IMG_H = 28
CAPACITY = 2000


def get_one_image(train):
   '''Randomly pick one image from training data
   Return: ndarray
   '''
   n = len(train)
   ind = np.random.randint(0, n)
   img_dir = train[ind]

   image = Image.open(img_dir)
   plt.imshow(image)
   image = image.resize([28, 28])
   image = np.array(image)
   return image


def evalute_one_pic(train_dir):

    train, train_label = input_file.get_files(train_dir)
    '''
    train_batch, train_label_batch = input_file.get_batch(train,
                                                          train_label,
                                                          IMG_W,
                                                          IMG_H,
                                                          BATCH_SIZE,
                                                          CAPACITY)
    '''
    image = get_one_image(train)

    #print(train_batch)
    with tf.Graph().as_default():
        image = tf.cast(image, tf.float32)
        image = tf.image.per_image_standardization(image)

        image = tf.reshape(image, [1, 28, 28, 3])

        logit = model.inference(image,BATCH_SIZE,N_CLASSES)
        logit = tf.nn.softmax(logit)

        #x = tf.placeholder(tf.float32,[1,28,28,3])
        saver = tf.train.Saver()
        #init = tf.initialize_all_variables()
        init = tf.initialize_all_variables()
        with tf.Session() as sess:
           sess.run(init)
           print("Reading checkpoints...")


           #ckpt = tf.train.get_checkpoint_state(log_dir)
           if 1:
               #global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
               saver.restore(sess,'.//logs//./file.ckpt')         #####这里超重要！！！！

               print('Loading success, global_step is %s')
           else:
               print('No checkpoint file found')

           #saver.restore(sess,'./logs//file.ckpt.data-00000-of-00001')
           prediction = sess.run(logit)
           print(prediction)
           max_index = np.argmax(prediction)

           print(max_index)


if __name__ == '__main__':
    evalute_one_pic(train_dir)

