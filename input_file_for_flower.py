import tensorflow as tf
import numpy as np
import os



# you need to change this to your data directory
CSV_NAME = "D:\\csdn\\flower_photos\\"
CSV_TEST = "csv_file_test.csv"


def get_files(file_dir):
    '''
    Args:
        file_dir: file directory
    Returns:
        list of images and labels
    '''

    daisy = []
    label_daisy = []

    dandelion = []
    label_dandelion = []

    roses = []
    label_roses = []

    sunflowers = []
    label_sunflowers = []

    tulips = []
    label_tulips = []

    for file in os.listdir(file_dir):
        name = file.split(sep='.')
        if name[0]=='tulips':
            tulips.append(file_dir + file)
            label_tulips.append(0)
        elif name[0] == 'sunflowers':
            sunflowers.append(file_dir+file)
            label_sunflowers.append(1)
        elif name[0] == 'roses':
            roses.append(file_dir+file)
            label_roses.append(2)
        elif name[0] == 'dandelion':
            dandelion.append(file_dir+file)
            label_dandelion.append(3)
        elif name[0] == "daisy":
            daisy.append(file_dir+file)
            label_daisy.append(4)

    print('There are %d sunflower\nThere are %d daisy\n %d ros'
          'es\n %d tulips \n %d danelion' %(len(sunflowers), len(daisy),len(roses),len(tulips),len(dandelion)))

    image_list = np.hstack((sunflowers,roses,dandelion,daisy,tulips))
    label_list = np.hstack((label_sunflowers,label_roses,label_dandelion,label_daisy,label_tulips))
    #label_list = label_sunflowers+label_roses+label_dandelion+label_daisy+label_tulips

    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)

    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]


    return image_list, label_list


#%%

def get_batch(image, label, image_W, image_H, batch_size, capacity):
    '''
    Args:
        image: list type
        label: list type
        image_W: image width
        image_H: image height
        batch_size: batch size
        capacity: the maximum elements in queue
    Returns:
        image_batch: 4D tensor [batch_size, width, height, 3], dtype=tf.float32
        label_batch: 1D tensor [batch_size], dtype=tf.int32
    '''

    image = tf.cast(image, tf.string)
    print(image,1)
    label = tf.cast(label, tf.int32)
    print(label,2)

    # make an input queue
    input_queue = tf.train.slice_input_producer([image, label])

    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)

    ######################################
    # data argumentation should go to here
    ######################################

    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)

    # if you want to test the generated batches of images, you might want to comment the following line.
    # 如果想看到正常的图片，请注释掉111行（标准化）和 126行（image_batch = tf.cast(image_batch, tf.float32)）
    # 训练时不要注释掉！
    image = tf.image.per_image_standardization(image)

    '''
    image_batch, label_batch = tf.train.batch([image, label],
                                                batch_size= batch_size,
                                                num_threads= 64,
                                                capacity = capacity)
    '''
    #you can also use shuffle_batch
    image_batch, label_batch = tf.train.shuffle_batch([image,label],
                                                      batch_size=batch_size,
                                                     num_threads=64,
                                                      capacity=capacity,
                                                      min_after_dequeue=capacity-1)

    label_batch = tf.reshape(label_batch, [batch_size])
    #print(image_batch[0])
    image_batch = tf.cast(image_batch, tf.float32)
    #print(image_batch)
    #print(image_batch,4)
    #image_batch = tf.reshape(image_batch,[-1,129792])
    #print(image_batch,5)
    #print(label_batch,4)
    #label_batch = tf.reshape(label_batch,[-1,5])
    #print(label_batch,5)
    return image_batch, label_batch


if __name__ == '__main__':
    BATCH_SIZE = 10
    CAPACITY = 1
    IMG_W = 208
    IMG_H = 208

    train_dir = CSV_NAME

    image_list, label_list = get_files(train_dir)
    #print(image_list,label_list)
    image_batch, label_batch = get_batch(image_list, label_list, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)


    with tf.Session() as sess:
        i = 0
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            while not coord.should_stop() and i<1:

                img, label = sess.run([image_batch, label_batch])
                print(label)
             #just test one batch

                for j in np.arange(BATCH_SIZE):
                    print('label: %d' %label[j])
                    #plt.imshow(img[j,:,:,:])
                    #plt.show()
                    #plt.close()
                i+=1

        except tf.errors.OutOfRangeError:
            print('done!')
        finally:
                coord.request_stop()
        coord.join(threads)


















#%% TEST
# To test the generated batches of images
# When training the model, DO comment the following codes




#import matplotlib.pyplot as plt
#
#BATCH_SIZE = 2
#CAPACITY = 256
#IMG_W = 208
#IMG_H = 208
#
#train_dir = '/home/kevin/tensorflow/cats_vs_dogs/data/train/'
#
#image_list, label_list = get_files(train_dir)
#image_batch, label_batch = get_batch(image_list, label_list, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
#
#with tf.Session() as sess:
#    i = 0
#    coord = tf.train.Coordinator()
#    threads = tf.train.start_queue_runners(coord=coord)
#
#    try:
#        while not coord.should_stop() and i<1:
#
#            img, label = sess.run([image_batch, label_batch])
#
#            # just test one batch
#            for j in np.arange(BATCH_SIZE):
#                print('label: %d' %label[j])
#                plt.imshow(img[j,:,:,:])
#                plt.show()
#            i+=1
#
#    except tf.errors.OutOfRangeError:
#        print('done!')
#    finally:
#        coord.request_stop()
#    coord.join(threads)