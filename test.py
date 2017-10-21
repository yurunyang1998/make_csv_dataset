import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import input_file_for_flower as input_data
import numpy as np
import model
def get_one_image(train):
   '''Randomly pick one image from training data
   Return: ndarray
   '''
   n = len(train)
   ind = np.random.randint(0, n)
   img_dir = train[ind]

   image = Image.open(img_dir)
   plt.imshow(image)
   image = image.resize([208, 208])
   image = np.array(image)
   return image

def evaluate_one_image():
   '''Test one image against the saved models and parameters
   '''

   # you need to change the directories to yours.
   train_dir = r"D:\\csdn\\flower_photos\\"
   train, train_label = input_data.get_files(train_dir)
   image_array = get_one_image(train)

   with tf.Graph().as_default():
       BATCH_SIZE = 1
       N_CLASSES = 5

       image = tf.cast(image_array, tf.float32)
       image = tf.image.per_image_standardization(image)
       image = tf.reshape(image, [1, 208, 208, 3])
       logit = model.inference(image, BATCH_SIZE, N_CLASSES)

       logit = tf.nn.softmax(logit)

       x = tf.placeholder(tf.float32, shape=[208, 208, 3])

       # you need to change the directories to yours.
       logs_train_dir = r'D:\catvsdog\logs\model.ckpt'

       saver = tf.train.Saver()
       init = tf.initialize_all_variables()
       with tf.Session() as sess:
           sess.run(init)
           print("Reading checkpoints...")
           ckpt = tf.train.get_checkpoint_state(logs_train_dir)
           if ckpt and ckpt.model_checkpoint_path:
               global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
               saver.restore(sess, ckpt.model_checkpoint_path)
               print('Loading success, global_step is %s' % global_step)
           else:
               print('No checkpoint file found')

           prediction = sess.run(logit, feed_dict={x: image_array})
           max_index = np.argmax(prediction)

           print(max_index)
if __name__ == '__main__':
    evaluate_one_image()