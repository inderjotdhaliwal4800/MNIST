#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.preprocessing.image import load_img
from ann_visualizer.visualize import ann_viz


# In[2]:


seed = 7
np.random.seed(seed)
(X_train, y_train), (X_test, y_test) = mnist.load_data()


# In[3]:


img_rows, img_cols = 28, 28

X_train = X_train.reshape(X_train.shape[0], img_cols, img_rows, 1)
X_test = X_test.reshape(X_test.shape[0], img_cols, img_rows, 1)


# In[4]:


X_train = X_train / 255
X_test = X_test / 255


# In[5]:


y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]


# In[6]:


def larger_model():
    
    model = Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=(img_cols, img_rows, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# In[7]:


model = larger_model()


# In[8]:


model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)


# In[9]:


scores = model.evaluate(X_test, y_test, verbose=0)
print("Large CNN Accuracy: %.2f%%" % (scores[1]*100))


# In[10]:


ann_viz(model, title='A CNN model to identify handwritten digits')


# In[11]:


img = load_img('im8.jpg')


# In[12]:


img.size


# In[13]:


img


# In[14]:


img = np.resize(img,(28,28))
img = np.reshape(img,[1,28,28, 1])


# In[15]:


pred= model.predict([img])


# In[16]:


pred


# In[17]:


val = np.argmax(pred)


# In[18]:


val


# In[ ]:




