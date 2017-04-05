#Importing
import numpy as np
from scipy import signal
from scipy import misc
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf

#read the image as Float data type
im=misc.imread("/resources/data/lena.png").astype(np.float)

#im=misc.imread("one.png").astype(np.float)

#Convert image to gray scale
grayim=np.dot(im[...,:3], [0.299, 0.587, 0.114])


#Plot the images
%matplotlib inline

plt.subplot(1, 2, 1)
plt.imshow(im)
plt.xlabel(" Float Image ")

plt.subplot(1, 2, 2)
plt.imshow(grayim, cmap=plt.get_cmap("gray"))
plt.xlabel(" Gray Scale Image ")

Image = np.expand_dims(np.expand_dims(grayim, 0), -1)

print Image.shape

img= tf.placeholder(tf.float32, [None,512,512,1])
print img.get_shape().as_list()

shape=[5,5,1,1]
weights =tf.Variable(tf.truncated_normal(shape, stddev=0.05))
print weights.get_shape().as_list()

ConOut = tf.nn.conv2d(input=img,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

ConOut2 = tf.nn.conv2d(input=img,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='VALID')

init = tf.global_variables_initializer()
sess= tf.Session()
sess.run(init)

result = sess.run(ConOut,feed_dict={img:Image})

result2 = sess.run(ConOut2,feed_dict={img:Image})


# for the result with 'SAME' Padding 

#reduce the dimension
vec = np.reshape(result, (1, -1));
# Reshape the image
image= np.reshape(vec,(512,512))

print image.shape


# for the result with 'VALID' Padding 

#reduce the dimension
vec2 = np.reshape(result2, (1, -1));
# Reshape the image
image2= np.reshape(vec2,(508,508))

print image2.shape

#Plot the images
%matplotlib inline

plt.subplot(1, 2, 1)
plt.imshow(image,cmap=plt.get_cmap("gray"))
plt.xlabel(" SAME Padding ")

plt.subplot(1, 2, 2)
plt.imshow(image2, cmap=plt.get_cmap("gray"))
plt.xlabel(" VALID Padding ")


def conv2d (X,W):
    return tf.nn.conv2d(input=X,filter=W,strides=[1, 1, 1, 1],padding='SAME')
    
 

def MaxPool (X):
    return tf.nn.max_pool(X, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


weights = {
	'W_conv1': tf.Variable(tf.random_normal([5, 5, 1, 32]))       
    }

biases = {
        'b_conv1': tf.Variable(tf.random_normal([32]))    
    }
     


conv1 = tf.nn.relu(conv2d(img, weights['W_conv1']) + biases['b_conv1'])

Mxpool = Maxpool (conv1)   
print conv1.get_shape().as_list()
print Mxpool.get_shape().as_list()


init = tf.global_variables_initializer()
sess= tf.Session()
sess.run(init)

Layer1 = sess.run(Mxpool,feed_dict={img:Image})

print Layer1.shape

vec = np.reshape(Layer1, (256,256,32));
print vec.shape

for i in range (32):
    
    image=vec[:,:,i]
    #print image
    #image *= 255.0/image.max() 
    #print image
    plt.imshow(image,cmap=plt.get_cmap("gray"))
    plt.xlabel( i , fontsize=20, color='red')
    plt.show()
    plt.close()
   
