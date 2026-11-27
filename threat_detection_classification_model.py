import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import albumentations as A
from PIL import Image

# Augmentation pipeline
augment = A.Compose([
    A.Resize(224, 224),
])

# Frame extraction from video
def extract_frames(video_path, max_frames=50, frame_rate=1):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while cap.isOpened() and count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if int(cap.get(1)) % frame_rate == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = augment(image=frame)['image']
            frame = frame / 255.0  # Normalize
            frames.append(frame)
            count += 1
    cap.release()
    return frames

# Load and preprocess images
def process_image_folder(folder_path):
    images, labels = [], []
    for label in os.listdir(folder_path):
        class_path = os.path.join(folder_path, label)
        for file in tqdm(os.listdir(class_path), desc=f"Loading {label}"):
            img_path = os.path.join(class_path, file)
            img = np.array(Image.open(img_path).convert("RGB"))
            img = augment(image=img)['image']
            img = img / 255.0
            images.append(img)
            labels.append(label)
    return images, labels

# Main preprocessing function
def preprocess_dataset(image_dir):
    all_images, all_labels = [], []

    # Process images
    img_data, img_labels = process_image_folder(image_dir)
    all_images.extend(img_data)
    all_labels.extend(img_labels)

    # Process videos
    # for video_file in tqdm(os.listdir(video_dir), desc="Processing videos"):
    #     label = video_file.split("_")[0]  # e.g. "fire_accident_001.mp4" → "fire"
    #     video_path = os.path.join(video_dir, video_file)
    #     frames = extract_frames(video_path)
    #     all_images.extend(frames)
    #     all_labels.extend([label] * len(frames))

    # Encode labels
    le = LabelEncoder()
    encoded_labels = le.fit_transform(all_labels)

    # Convert to arrays
    X = np.array(all_images, dtype=np.float32)
    y = np.array(encoded_labels)

    return X, y, le

X, y, label_encoder = preprocess_dataset("C:/Users/jlrab/OneDrive/Documents/threat_detection_dataset/images")

def split_and_save(X, y, out_dir="threat_detection_system"):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    np.save(os.path.join(out_dir, "X_train.npy"), X_train)
    np.save(os.path.join(out_dir, "X_test.npy"), X_test)
    np.save(os.path.join(out_dir, "y_train.npy"), y_train)
    np.save(os.path.join(out_dir, "y_test.npy"), y_test)

split_and_save(X, y)
print("Classes:", label_encoder.classes_)



# import keras
# from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Activation
# from keras.layers import AveragePooling2D, Input, Flatten
# from keras.optimizers import Adam
# from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
# from keras.callbacks import ReduceLROnPlateau
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from keras.regularizers import l2
# from keras import backend as K
# from keras.models import Model
# from keras.datasets import cifar10
# import numpy as np
# import os

# batch_size = 32
# epochs = 200
# data_augmentation = True
# subtract_pixel_mean = True
# n = 3
# version = 1

# if version == 1:
#     depth = n * 6 + 2
# elif version == 2:
#     depth = n * 9 + 2

# model_type = 'ResNet % dv % d' % (depth, version)

# x_train = np.load("threat_detection_system/X_train.npy")
# x_test = np.load("threat_detection_system/X_test.npy")
# y_train = np.load("threat_detection_system/y_train.npy")
# y_test = np.load("threat_detection_system/y_test.npy")

# num_classes = len(np.unique(y_train))

# input_shape = x_train.shape[1:]

# if subtract_pixel_mean:
#     x_train_mean = np.mean(x_train, axis=0)
#     x_train -= x_train_mean
#     x_test -= x_train_mean

# print('x_train shape:', x_train.shape)
# print(x_train.shape[0], 'train samples')
# print(x_test.shape[0], 'test samples')
# print('y_train shape:', y_train.shape)

# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)


# print('y_train shape:', y_train.shape)

# def lr_schedule(epoch):
#     lr = 1e-3
#     if epoch > 180:
#         lr *= 0.5e-3
#     elif epoch > 160:
#         lr *= 1e-3
#     elif epoch > 120:
#         lr *= 1e-2
#     elif epoch > 80:
#         lr *= 1e-1
#     print('Learning rate: ', lr)
#     return lr

# def resnet_layer(inputs,
#                  num_filters=32,
#                  kernel_size=3,
#                  strides=1,
#                  activation='relu',
#                  batch_normalization=True,
#                  conv_first=True):
#     conv = Conv2D(num_filters,
#                   kernel_size=kernel_size,
#                   strides=strides,
#                   padding='same',
#                   kernel_initializer='he_normal',
#                   kernel_regularizer=l2(1e-4))

#     x = inputs
#     if conv_first:
#         x = conv(x)
#         if batch_normalization:
#             x = BatchNormalization()(x)
#         if activation is not None:
#             x = Activation(activation)(x)
#     else:
#         if batch_normalization:
#             x = BatchNormalization()(x)
#         if activation is not None:
#             x = Activation(activation)(x)
#         x = conv(x)

#     return x

# def resnet_v1(input_shape, depth, num_classes=num_classes): 
#     if (depth - 2) % 6 != 0: 
#         raise ValueError('depth should be 6n + 2 (eg 20, 32, 44 in [a])') 

#     num_filters = 32
#     num_res_blocks = int((depth - 2) / 6) 

#     inputs = Input(shape=input_shape) 
#     x = resnet_layer(inputs=inputs) 

#     for stack in range(3): 
#         for res_block in range(num_res_blocks): 
#             strides = 1
#             if stack > 0 and res_block == 0: 
#                 strides = 2 
#             y = resnet_layer(inputs=x, 
#                              num_filters=num_filters, 
#                              strides=strides) 
#             y = resnet_layer(inputs=y, 
#                              num_filters=num_filters, 
#                              activation=None) 
#             if stack > 0 and res_block == 0: 
#                 x = resnet_layer(inputs=x, 
#                                  num_filters=num_filters, 
#                                  kernel_size=1, 
#                                  strides=strides, 
#                                  activation=None, 
#                                  batch_normalization=False) 
#             x = keras.layers.add([x, y]) 
#             x = Activation('relu')(x) 
#         num_filters *= 2

#     x = AveragePooling2D(pool_size=8)(x) 
#     y = Flatten()(x) 
#     outputs = Dense(num_classes, 
#                     activation='softmax', 
#                     kernel_initializer='he_normal')(y) 

#     model = Model(inputs=inputs, outputs=outputs) 
#     return model

# def resnet_v2(input_shape, depth, num_classes=num_classes):
#     if (depth - 2) % 9 != 0:
#         raise ValueError('depth should be 9n + 2 (eg 56 or 110 in [b])')

#     num_filters_in = 32
#     num_res_blocks = int((depth - 2) / 9)

#     inputs = Input(shape=input_shape)
#     x = resnet_layer(inputs=inputs,
#                      num_filters=num_filters_in,
#                      conv_first=True)

#     for stage in range(3):
#         for res_block in range(num_res_blocks):
#             activation = 'relu'
#             batch_normalization = True
#             strides = 1
#             if stage == 0:
#                 num_filters_out = num_filters_in * 4
#                 if res_block == 0:
#                     activation = None
#                     batch_normalization = False
#             else:
#                 num_filters_out = num_filters_in * 2
#                 if res_block == 0:
#                     strides = 2

#             y = resnet_layer(inputs=x,
#                              num_filters=num_filters_in,
#                              kernel_size=1,
#                              strides=strides,
#                              activation=activation,
#                              batch_normalization=batch_normalization,
#                              conv_first=False)
#             y = resnet_layer(inputs=y,
#                              num_filters=num_filters_in,
#                              conv_first=False)
#             y = resnet_layer(inputs=y,
#                              num_filters=num_filters_out,
#                              kernel_size=1,
#                              conv_first=False)
#             if res_block == 0:
#                 x = resnet_layer(inputs=x,
#                                  num_filters=num_filters_out,
#                                  kernel_size=1,
#                                  strides=strides,
#                                  activation=None,
#                                  batch_normalization=False)
#             x = keras.layers.add([x, y])

#         num_filters_in = num_filters_out

#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#     x = AveragePooling2D(pool_size=8)(x)
#     y = Flatten()(x)
#     outputs = Dense(num_classes,
#                     activation='softmax',
#                     kernel_initializer='he_normal')(y)

#     model = Model(inputs=inputs, outputs=outputs)
#     return model

# if version == 2:
#     model = resnet_v2(input_shape=input_shape, depth=depth)
# else:
#     model = resnet_v1(input_shape=input_shape, depth=depth)

# model.compile(loss='categorical_crossentropy',
#               optimizer=Adam(learning_rate=lr_schedule(0)),
#               metrics=['accuracy'])
# model.summary()
# print(model_type)

# save_dir = os.path.join('threat_detection_system', 'saved_models')
# model_name = 'threat_detection_system_%s_model.{epoch:03d}.keras' % model_type
# if not os.path.isdir(save_dir):
#     os.makedirs(save_dir)
# filepath = os.path.join(save_dir, model_name)

# checkpoint = ModelCheckpoint(filepath=filepath,
#                               monitor='val_acc',
#                               verbose=1,
#                               save_best_only=True)

# lr_scheduler = LearningRateScheduler(lr_schedule)

# lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
#                                cooldown=0,
#                                patience=5,
#                                min_lr=0.5e-6)

# callbacks = [checkpoint, lr_reducer, lr_scheduler]

# if not data_augmentation:
#     print('Not using data augmentation.')
#     model.fit(x_train, y_train,
#               batch_size=batch_size,
#               epochs=epochs,
#               validation_data=(x_test, y_test),
#               shuffle=True,
#               callbacks=callbacks)
# else:
#     print('Using real-time data augmentation.')
#     # Complete the ImageDataGenerator
#     datagen = ImageDataGenerator(
#         featurewise_center=False,
#         samplewise_center=False,
#         zca_whitening=False,
#         rotation_range=20,
#         width_shift_range=0.2,
#         height_shift_range=0.2,
#         horizontal_flip=True,
#         fill_mode='nearest'
#     )
    
#     # Fit the generator on the training data
#     datagen.fit(x_train)

#     # Use the generator for training
#     model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),
#               steps_per_epoch=x_train.shape[0] // batch_size,
#               epochs=epochs,
#               validation_data=(x_test, y_test),
#               callbacks=callbacks)