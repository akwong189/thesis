# based on code from https://github.com/zhixuhao/unet/blob/master/model.py

# import cv2
import tensorflow as tf
# import matplotlib as mpl
# from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
# import imageio
from sklearn.model_selection import train_test_split

# loss function from densedepth https://github.com/ialhashim/DenseDepth
import tensorflow.keras.backend as K

def depth_loss_function(y_true, y_pred, theta=0.1, maxDepthVal=1000.0/10.0):

    # Point-wise depth
    l_depth = K.mean(K.abs(y_pred - y_true), axis=-1)

    # Edges
    dy_true, dx_true = tf.image.image_gradients(y_true)
    dy_pred, dx_pred = tf.image.image_gradients(y_pred)
    l_edges = K.mean(K.abs(dy_pred - dy_true) + K.abs(dx_pred - dx_true), axis=-1)

    # Structural similarity (SSIM) index
    l_ssim = K.clip((1 - tf.image.ssim(y_true, y_pred, maxDepthVal)) * 0.5, 0, 1)

    # Weights
    w1 = 1.0
    w2 = 1.0
    w3 = theta

    return (w1 * l_ssim) + (w2 * K.mean(l_edges)) + (w3 * K.mean(l_depth))

PATH="/data3/awong"

train = pd.read_csv(f"{PATH}/data/nyu2_train.csv", header=None, names=["Image", "Depth"])
test = pd.read_csv(f"{PATH}/data/nyu2_test.csv", header=None, names=["Image", "Depth"])


# Train/Test loader
train_image_path = f"{PATH}/" + train.Image
train_depth_path = f"{PATH}/" + train.Depth

# add validation
train_image_path, train_depth_path, validation_image_path, validation_depth_path = train_test_split(train_image_path, train_depth_path, test_size=0.5)

test_image_path = f"{PATH}/" + test.Image
test_depth_path = f"{PATH}/" + test.Depth

# auto load into the GPU for preprocessing (faster than previous method and allows for larger images)
AUTO = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32

@tf.function
def preprocess(image_path, depth_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    # image = tf.image.resize(image, [240,320])
    # image = tf.image.convert_image_dtype(image, tf.float16)

    depth = tf.io.read_file(depth_path)
    depth = tf.image.decode_png(depth, channels=1)
    depth = tf.image.resize(depth, [240,320])
    # depth = tf.image.convert_image_dtype(depth, tf.float16)

    return image, depth

trainloader = tf.data.Dataset.from_tensor_slices((train_image_path, train_depth_path))
testloader = tf.data.Dataset.from_tensor_slices((test_image_path, test_depth_path))
validationloader = tf.data.Dataset.from_tensor_slices((validation_image_path, validation_depth_path))

trainloader = (
    trainloader
    .map(preprocess, num_parallel_calls=AUTO)
    .shuffle(1024)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

validationloader = (
    validationloader
    .map(preprocess, num_parallel_calls=AUTO)
    .shuffle(1024)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

testloader = (
    testloader
    .map(preprocess, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

# model
def conv(input, layer):
    conv = tf.keras.layers.Conv2D(layer, 3, padding = "same")(input)
    conv = tf.keras.layers.BatchNormalization()(conv)
    conv = tf.keras.layers.ReLU()(conv)

    return conv

def upconv(input, layer):
    up = tf.keras.layers.UpSampling2D(size=(2,2))(input)
    conv = tf.keras.layers.Conv2D(layer, 2, padding = "same")(up)

    return conv

def get_model(input_size = (240, 320, 3)):
    input = tf.keras.Input(shape=input_size)

    # DOWN SAMPLING
    conv1 = conv(input, 64)
    conv1 = conv(conv1, 64) # copied and cropped

    maxpool1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(conv1)

    conv2 = conv(maxpool1, 128)
    conv2 = conv(conv2, 128) # copied and cropped

    maxpool2 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(conv2)

    conv3 = conv(maxpool2, 256)
    conv3 = conv(conv3, 256) # copied and cropped

    maxpool3 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(conv3)

    conv4 = conv(maxpool3, 512)
    conv4 = conv(conv4, 512) # copied and cropped

    maxpool4 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(conv4)

    conv5 = conv(maxpool4, 1024)
    conv5 = conv(conv5, 1024)

    # UPSAMPLING
    #1024 -> 512
    up1 = upconv(conv5, 512)
    cat1 = tf.keras.layers.concatenate([conv4, up1], axis = 3)

    conv6 = tf.keras.layers.Conv2D(512, 3, activation="relu", padding="same")(cat1)
    conv6 = tf.keras.layers.Conv2D(512, 3, activation="relu", padding="same")(conv6)

    # 512 -> 256
    up2 = upconv(conv6, 256)
    cat2 = tf.keras.layers.concatenate([conv3, up2], axis = 3)

    conv7 = tf.keras.layers.Conv2D(256, 3, activation="relu", padding="same")(cat2)
    conv7 = tf.keras.layers.Conv2D(256, 3, activation="relu", padding="same")(conv7)

    # 256 -> 128
    up3 = upconv(conv7, 128)
    cat3 = tf.keras.layers.concatenate([conv2, up3], axis = 3)

    conv8 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same")(cat3)
    conv8 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same")(conv8)

    # 128 -> 64
    up4 = upconv(conv8, 64)
    cat4 = tf.keras.layers.concatenate([conv1, up4], axis = 3)

    conv9 = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(cat4)
    conv9 = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(conv9)
    conv9 = tf.keras.layers.Conv2D(2, 3, activation="relu", padding="same")(conv9)

    conv10 = tf.keras.layers.Conv2D(1, 1, activation="sigmoid")(conv9)

    model = tf.keras.Model(input, outputs = conv10)

    return model

def model_2():
    def upsampling(input_tensor, n_filters, concat_layer):
        '''
        Constitutes the block of Decoder
        '''
        # Bilinear 2x upsampling layer
        x = tf.keras.layers.UpSampling2D(size=(2,2), interpolation='bilinear')(input_tensor)
        # concatenation with encoder block 
        x = tf.keras.layers.concatenate([x,concat_layer])
        # decreasing the depth filters by half
        x = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=(3,3), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=(3,3), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        return x

    pools = ['pool3_pool', 'pool2_pool', 'pool1', 'conv1/relu']

    encoder = tf.keras.applications.densenet.DenseNet121(include_top=False, weights='imagenet', input_shape=(480,640,3))
    x = encoder.output
    # decoder blocks linked with corresponding encoder blocks
    bneck = tf.keras.layers.Conv2D(filters=1024, kernel_size=(1,1), padding='same')(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(bneck)
    x = upsampling(bneck, 1024, encoder.get_layer(pools[0]).output)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = upsampling(x, 416, encoder.get_layer(pools[1]).output)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = upsampling(x, 208, encoder.get_layer(pools[2]).output)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = upsampling(x, 104, encoder.get_layer(pools[3]).output)
    x = tf.keras.layers.Conv2D(filters=1, kernel_size=(3,3), padding='same')(x)

    model = tf.keras.Model(inputs=encoder.input, outputs=x)

    return model

model = get_model()
# model.summary()

base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=depth_loss_function,
              metrics=['accuracy'])

# early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_label_loss', mode='min', patience=10, restore_best_weights=True)

model.fit(
  trainloader,
  validation_data=validationloader,
  epochs=20,
#   callbacks=[early_stop],
)

model.evaluate(testloader)
model.save('unet-mae.h5')
