from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D,ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense, Input
from keras.utils import to_categorical
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input, VGG16
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
import numpy as np

img_width, img_height = 150, 150
Weights_PATH = r'E:\vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

# LOAD VGG16
input_tensor = Input(shape=input_shape)
model = VGG16(weights=Weights_PATH,
              include_top=False,
              input_tensor=input_tensor)

top_model = Sequential()
top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(2, activation='softmax'))
# top_model.load_weights(top_model_weights_path)


# CREATE AN "REAL" MODEL FROM VGG16
# BY COPYING ALL THE LAYERS OF VGG16
new_model = Sequential()
for l in model.layers:
    new_model.add(l)
new_model.add(top_model)

# from keras.optimizers import SGD
# for layer in new_model.layers[:4]:
#     layer.trainable = False
# #compile the model with a SGD/momentum optimizer
# # and a very slow learning rate.
# new_model.compile(loss='categorical_crossentropy',
#               optimizer=SGD(lr=1e-4, momentum=0.9),
#               metrics=['accuracy'])

# train_data_dir = r'E:\dogs-vs-cats\train'
# validation_data_dir = r'E:\dogs-vs-cats\validation'
# nb_train_samples = 10835
# nb_validation_samples = 4000
# epochs = 100
# batch_size = 100
#
#
# # this is the augmentation configuration we will use for training
# train_datagen = ImageDataGenerator(
#     rescale=1. / 255,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True)
#
# # this is the augmentation configuration we will use for testing:
# # only rescaling
# test_datagen = ImageDataGenerator(rescale=1. / 255)
#
# train_generator = train_datagen.flow_from_directory(
#     train_data_dir,
#     target_size=(img_width, img_height),
#     batch_size=batch_size,
#     class_mode='categorical')
#
# validation_generator = test_datagen.flow_from_directory(
#     validation_data_dir,
#     target_size=(img_width, img_height),
#     batch_size=batch_size,
#     class_mode='categorical')
#
# Log_Save_Path = r"E:\广州_AI\广州AI课件及作业\6.卷积神经网络\model\11.8\log"
# tensorboard = TensorBoard(log_dir=Log_Save_Path)
#
# Best_Model_Save_Path = r"E:\广州_AI\广州AI课件及作业\6.卷积神经网络\model\11.8\BestModel\weights_best.hdf5"
# checkpoint = ModelCheckpoint(Best_Model_Save_Path, monitor='val_accuracy', verbose=1, save_best_only=True,mode='max')
#
#
# callback_lists = [tensorboard, checkpoint]  # 因为callback是list型,必须转化为list
#
# new_model.fit_generator(
#     train_generator,
#     steps_per_epoch=nb_train_samples // batch_size,
#     epochs=epochs,
#     validation_data=validation_generator,
#     validation_steps=nb_validation_samples // batch_size,
#     shuffle='True',
#     callbacks=callback_lists)

import cv2
img = cv2.resize(cv2.imread(r'E:\dogs-vs-cats\test\28.jpg'), (img_width, img_height)).astype(np.float32)
# img[:,:,0] -= 103.939
# img[:,:,1] -= 116.779
# img[:,:,2] -= 123.68
#img = img.transpose((2,0,1))
x = img_to_array(img)

x = np.expand_dims(x, axis=0)

#x = preprocess_input(x)
new_model.load_weights(r"E:\广州_AI\广州AI课件及作业\6.卷积神经网络\model\11.8\BestModel\weights_best.hdf5")
score = new_model.predict(x)


print(score)