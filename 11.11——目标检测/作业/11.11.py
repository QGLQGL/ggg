# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # 实验一
# 自定义Inception网络，并在Kaggle猫/狗数据集上进行训练和测试
# ![image.png](attachment:image.png)
# %% [markdown]
# ## 1.加载keras模块

# %%
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Conv2D, AveragePooling2D,Input,BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense,Concatenate
from keras.utils import to_categorical
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras import backend as K
import numpy as np

# %% [markdown]
# ### 定义Inception网络结构
# 
# 

# %%
#声明Input layer并设置input_shape    
img_width, img_height = 50, 50
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
    bn_axis = 1
else:
    input_shape = (img_width, img_height, 3)
    bn_axis = 3

x = Input(shape=input_shape)
  
#以函数形式定义inception module的分支，可适当比上图inception结构减少滤波器个数
#branch 1
branch1_out = Conv2D(16, (1, 1),
    padding="same",
    use_bias=False)(x)
branch1_out = BatchNormalization(axis=bn_axis)(branch1_out)
branch1_out = Activation('relu')(branch1_out)
    
    
#branch 2   
branch2_out = Conv2D(
         16, (1, 1),
         padding="same",
         use_bias=False)(x)
branch2_out = BatchNormalization(axis=bn_axis)(branch2_out)
branch2_out = Activation('relu')(branch2_out)    
branch2_out = Conv2D(
         48, (3, 3),
         padding="same",
         use_bias=False)(branch2_out)
branch2_out = BatchNormalization(axis=bn_axis)(branch2_out)
branch2_out = Activation('relu')(branch2_out) 


#branch 3
branch3_out = Conv2D(
       16, (1, 1),
        padding="same",
        use_bias=False)(x)
branch3_out = BatchNormalization(axis=bn_axis)(branch3_out)
branch3_out = Activation('relu')(branch3_out)
branch3_out = Conv2D(
         24, (5, 5),
         padding="same",
         use_bias=False)(branch3_out)
branch3_out = BatchNormalization(axis=bn_axis)(branch3_out)
branch3_out = Activation('relu')(branch3_out) 

#branch 4
branch4_out = AveragePooling2D(
        pool_size=(3, 3),strides=(1, 1), padding='same', data_format=K.image_data_format())(x)
branch4_out = Conv2D(
         16, (1, 1),
         padding="same",
         use_bias=False)(branch4_out)
branch4_out = BatchNormalization(axis=bn_axis)(branch4_out)
branch4_out = Activation('relu')(branch4_out) 

#concatenate layer
out = Concatenate(axis=bn_axis)([branch1_out, branch2_out, branch3_out, branch4_out])
out = Conv2D(
         16, (1, 1),
         padding="same",
         use_bias=False)(out)
#fully connected layer
out = Flatten()(out)
out = Dense(48, activation='relu')(out)
#output layer
out = Dense(1, activation='sigmoid')(out)

model = Model(x, out)

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

#以函数形式定义inception module的concatenate layer


#为减少网络规模，可使用1*1 conv layer压缩feature map深度


#定义fully connected layer

#定义output layer


#调用Model函数声明model

#编译整个网络，声明loss函数和优化器、metrics

# %% [markdown]
# ### 查看model架构
# 
# 

# %%
model.summary()

# %% [markdown]
# ### 定义ImageDataGenerator
# 

# %%
train_data_dir = r'G:\BaiduNetdiskDownload\百度云文件下载\python培训课课件等\11.06 机器视觉1\作业\dogs-vs-cats\train'
validation_data_dir = r'G:\BaiduNetdiskDownload\百度云文件下载\python培训课课件等\11.06 机器视觉1\作业\dogs-vs-cats\validation'
nb_train_samples = 10835
nb_validation_samples = 4000
epochs = 5
batch_size = 10


# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

#调用flow_from_directory函数读取training数据和validation数据，注意
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

# %% [markdown]
# ### 训练模型
# 
# 

# %%
#调用fit_generator函数
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)
# %% [markdown]
# ### 使用训练后模型预测图像
# 
# 
# 
# 

# %%
from cv2 import cv2 as cv 
img = cv.resize(cv.imread(r'G:\BaiduNetdiskDownload\百度云文件下载\python培训课课件等\11.06 机器视觉1\作业\dogs-vs-cats\test\7.jpg'), (img_width, img_height)).astype(np.float32)
# img[:,:,0] -= 103.939
# img[:,:,1] -= 116.779
# img[:,:,2] -= 123.68
#img = img.transpose((2,0,1))
x = img_to_array(img)

x = np.expand_dims(x, axis=0)

#x = preprocess_input(x)

score = model.predict(x)


print(score)

