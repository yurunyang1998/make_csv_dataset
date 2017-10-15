import tensorflow as tf
import cv2
import numpy as np
import model


log_dir = r'D:\csdn\logs'
IMG_DIR = r'D:\csdn\flower_photos\daisy.0 (26).jpg'


def get_img(img_dir):
    img = cv2.imread(img_dir,flags=cv2.IMREAD_ANYCOLOR)
    #cv2.imshow('pic',img)
    img=cv2.resize(img,(28,28))
    #cv2.imshow('pic',img)
    np.array(img)
    return img



def evalute_one_pic(img_dir):
    img = get_img(img_dir)
    init = tf.initialize_all_variables()
    with tf.Graph().as_default():
        BATCH_SIZE = 1
        N_CLASSES = 5
        img = tf.cast(img,tf.float32)
        img = tf.reshape(img,[1,28,28,3])
        logit = model.inference(img,BATCH_SIZE,N_CLASSES)
        logit = tf.nn.softmax(logit)

        x = tf.placeholder(tf.float32,[1,28,28,3])
        #saver = tf.train.Saver()
        #init = tf.initialize_all_variables()
        init = tf.initialize_all_variables()
        with tf.Session() as sess:
           sess.run(init)
           print("Reading checkpoints...")
           ckpt = tf.train.get_checkpoint_state(log_dir)
           if ckpt and ckpt.model_checkpoint_path:
               global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
               saver.restore(sess, ckpt.model_checkpoint_path)
               print('Loading success, global_step is %s' % global_step)
           else:
               print('No checkpoint file found')

           prediction = sess.run(logit)
           max_index = np.argmax(prediction)

           print(max_index)


if __name__ == '__main__':
    evalute_one_pic(IMG_DIR)

