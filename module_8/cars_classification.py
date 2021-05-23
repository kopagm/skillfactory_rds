import csv
import glob
import os
import pickle
import subprocess
import sys
import tarfile
import zipfile
from datetime import datetime

from pandas.core.indexes import base

# import efficientnet.tfkeras as efn
import keras
import matplotlib.pyplot as plt
# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import PIL
import scipy.io
import seaborn as sns
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import tensorflow.keras.callbacks as C
import tensorflow.keras.layers as L
import tensorflow.keras.models as M
# from kaggle.api.kaggle_api_extended import KaggleApi
from keras import *
from keras.layers import *
from keras.models import load_model
from PIL import ImageFilter, ImageOps
from skimage import io
from sklearn.model_selection import train_test_split
#from tensorflow.keras.regularizers import l2
from tensorflow.keras import optimizers
from tensorflow.keras.applications import EfficientNetB6, Xception
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.callbacks import (Callback, EarlyStopping,
                                        LearningRateScheduler, ModelCheckpoint)
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#from tf.keras.applications.inception_v3 import Inception_v3


EPOCHS = 10  # эпох на обучение
# BATCH_SIZE = 64  # Xception # уменьшаем batch если сеть большая, иначе не влезет в память на GPU
BATCH_SIZE = 16  # EfficientNetB6
LR = 1e-4
VAL_SPLIT = 0.15  # сколько данных выделяем на тест = 15%

CLASS_NUM = 10  # количество классов в нашей задаче
IMG_SIZE = 224  # какого размера подаем изображения в сеть
# IMG_SIZE = 64  # какого размера подаем изображения в сеть

IMG_CHANNELS = 3   # у RGB 3 канала
input_shape = (IMG_SIZE, IMG_SIZE, IMG_CHANNELS)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
PYTHONHASHSEED = 0

# kaggle
DATA_PATH_KAGGLE = '../input/'
PATH_KAGGLE = '../car/'  # рабочая директория
# colab
DATA_PATH_COLAB = './'
PATH_COLAB = './'  # рабочая директория

IS_ENV_COLAB = 'google.colab' in sys.modules

if IS_ENV_COLAB:
    DATA_PATH = DATA_PATH_COLAB
    PATH = PATH_COLAB  # рабочая директория
else:
    DATA_PATH = DATA_PATH_KAGGLE
    PATH = PATH_KAGGLE  # рабочая директория


def download_competition_data():
    """
    dowload data in colab from kaggle
    """
    subprocess.run(['mkdir', '-p', '/root/.kaggle'], capture_output=True)
    subprocess.run(['cp', '/content/drive/MyDrive/Colab Notebooks/kaggle/kaggle.json',
                    '/root/.kaggle'], capture_output=True)
    subprocess.run(['kaggle', 'competitions', 'download', '-c',
                    'sf-dl-car-classification'], capture_output=True)


def unzip_data(path_input='.', path_output='.'):
    # unzip data
    if not all([dir in os.listdir() for dir in ['train', 'test_upload']]):
        for data_zip in ['train.zip', 'test.zip']:
            with zipfile.ZipFile(f'{path_input}{data_zip}', 'r') as z:
                z.extractall(path_output)


def load_data():
    """
    load DataFrames from files (train_df, sample_submission)
    """
    def load_from_files(path=''):
        train_df = pd.read_csv(f'{path}train.csv')
        sample_submission = pd.read_csv(f'{path}sample-submission.csv')
        return (train_df, sample_submission)

    if IS_ENV_COLAB:
        # running in colab
        download_competition_data()
    # else:
    #     # running in kaggel
    #     os.makedirs(PATH, exist_ok=False)
    unzip_data(DATA_PATH, PATH)
    return load_from_files(DATA_PATH)


def predict_submission(model, generator, name='new'):
    predictions = model.predict(generator, verbose=1)
    predictions = np.argmax(predictions, axis=-1)  # multiple categories
    label_map = (train_generator.class_indices)
    label_map = dict((v, k) for k, v in label_map.items())  # flip k,v
    predictions = [label_map[k] for k in predictions]

    filenames_with_dir = sub_generator.filenames
    submission = pd.DataFrame(
        {'Id': filenames_with_dir, 'Category': predictions}, columns=['Id', 'Category'])
    submission['Id'] = submission['Id'].replace('test_upload/', '')
    # submission.to_csv('submission.csv', index=False)
    submission.to_csv(f'{name}_submission.csv', index=False)
    if IS_ENV_COLAB:
        subprocess.run(['cp', f'{name}_submission.csv',
                        '/content/drive/MyDrive/Colab Notebooks/kaggle/models'], capture_output=True)


def predict_tta(model, generator, name='new', tta_steps=10):
    """
    from flowers
    in process
    """
    predictions = []

    for i in range(tta_steps):
        preds = model.predict(generator, verbose=1)
        predictions.append(preds)

    pred = np.mean(predictions, axis=0)

    predictions = np.argmax(predictions, axis=-1)  # multiple categories
    label_map = (train_generator.class_indices)
    label_map = dict((v, k) for k, v in label_map.items())  # flip k,v
    predictions = [label_map[k] for k in predictions]

    filenames_with_dir = sub_generator.filenames
    submission = pd.DataFrame(
        {'Id': filenames_with_dir, 'Category': predictions}, columns=['Id', 'Category'])
    submission['Id'] = submission['Id'].replace('test_upload/', '')
    # submission.to_csv('submission.csv', index=False)
    submission.to_csv(f'{name}_submission.csv', index=False)
    if IS_ENV_COLAB:
        subprocess.run(['cp', f'{name}_submission.csv',
                        '/content/drive/MyDrive/Colab Notebooks/kaggle/models'], capture_output=True)


def imshow(image_RGB):
    io.imshow(image_RGB)
    io.show()


def plot_images():
    print('Пример картинок (random sample)')
    plt.figure(figsize=(12, 8))

    random_image = train_df.sample(n=9)
    # random_image_paths = random_image['Id'].values
    # random_image_cat = random_image['Category'].values
    for index, (id, cat) in enumerate(zip(random_image['Id'], random_image['Category'])):
        # for index, path in enumerate(random_image_paths):
        im = PIL.Image.open(f'{PATH}train/{cat}/{id}')
        plt.subplot(3, 3, index+1)
        plt.imshow(im)
        plt.title('Class: '+str(cat))
        plt.axis('off')
    plt.show()


def plot_im_size(df):
    size = [PIL.Image.open(f'{PATH}train/{cat}/{id}').size for id,
            cat in zip(df['Id'], df['Category'])]
    s_x = list(map(lambda x: x[0], size))
    s_y = list(map(lambda x: x[1], size))
    plt.scatter(s_x, s_y, s=5)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Image resolutions')
    plt.show()
    # pd.value_counts(size)
    # print("Наиболее распространенные разрешения")
    # print(pd.value_counts(size).head(10))


def plot_images_from_generator(generator):
    x, y = generator.next()
    print('Пример картинок из train_generator')
    plt.figure(figsize=(12, 8))

    for i in range(0, BATCH_SIZE):
        image = x[i]
        plt.subplot(3, 3, i+1)
        plt.imshow(image)
        # plt.title('Class: ')
        # plt.axis('off')
    plt.show()


def plot_history(history, model_name='model', score=''):
    now = datetime.now().strftime("%d_%m_%Y_%H:%M:%S")

    plt.figure(figsize=(10, 5))
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'b', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title(f'Accuracy {model_name}. {score}')
    plt.legend()
    plt.savefig(f'accuracy_{model_name}_{now}.png', dpi=600)

    # plt.figure()
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(f'Loss {model_name}')
    plt.legend()
    # plt.savefig(f'loss_{model_name}_{now}.png', dpi=600)

    plt.show()


def load_model():
    subprocess.run(['cp', '/content/drive/MyDrive/Colab Notebooks/kaggle/models/best_model_3.hdf5',
                    'best_model.hdf5'], capture_output=True)
    model = tf.keras.models.load_model('best_model.hdf5')
    return model


def print_layers(model):
    for layer in model.layers:
        print(layer, layer.trainable)


def unfreeze_model(model, ratio=1):
    '''
    Unfreeze model layers.
    ratio - [0,1], 1 - all layers trainable.
    '''
    if 0 < ratio <= 1:
        model.trainable = True
        fine_tune_at = round(len(model.layers) * ratio)
        for layer in model.layers[:-fine_tune_at]:
            layer.trainable = False
    else:
        model.trainable = False


def compile_model(model, lr=LR):
    '''
    Compile model
    '''
    model.compile(loss="categorical_crossentropy",
                  optimizer=optimizers.Adam(lr),
                  metrics=["accuracy"])


def models_efn_base(input_shape, lr=1e-4):
    '''
    EfficientNetB6 +batchnorm
    EfficientNet models expect their inputs to be
    float tensors of pixels with values in the [0-255] range.
    '''

    base_model = EfficientNetB6(weights='imagenet',
                                include_top=False,
                                input_shape=input_shape,
                                pooling='avg')

    # base_model.trainable = False

    x = base_model.output
    # let's add a fully-connected layer
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    # and a logistic layer -- let's say we have 10 classes
    predictions = Dense(CLASS_NUM, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input,
                  outputs=predictions,
                  name=f'EfficientNetB6')
    model.compile(loss="categorical_crossentropy",
                  optimizer=optimizers.Adam(lr),
                  metrics=["accuracy"])

    return model, base_model


def models_xception_base(input_shape, lr=LR):
    '''
    Xception new
    For Xception, call
    tf.keras.applications.xception.preprocess_input 
    on your inputs before passing them to the 
    model.xception.preprocess_input will scale input 
    pixels between -1 and 1.
    '''
    base_model = Xception(weights='imagenet',
                          include_top=False,
                          input_shape=input_shape,
                          pooling='avg')
    # unfreeze_model(base_model, ratio=unfreeze_ratio)

    inputs = keras.Input(shape=input_shape)
    x = tf.cast(inputs, tf.float32)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    # base_model.summary()

    # let's add a fully-connected layer
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    # and a logistic layer -- let's say we have 10 classes
    predictions = Dense(CLASS_NUM, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=inputs,
                  outputs=predictions,
                  name=f'Xception')
    model.compile(loss="categorical_crossentropy",
                  optimizer=optimizers.Adam(lr), metrics=["accuracy"])

    # model.summary()
    return model, base_model


def fine_tune_fit(model_fun, input_shape, steps=[]):
    # step - (lr, unfreeze_ratio, epochs)
    # steps = [(1e-3, 0, 10),
    #          (1e-4, 0, 5),
    #          (1e-4, 0.25, 5),
    #          (1e-4, 0.5, 5),
    #          (1e-5, 1, 5)]

    model, base_model = model_fun(input_shape)

    for step, (lr, ratio, epochs) in enumerate(steps):

        unfreeze_model(base_model, ratio)
        compile_model(model, lr)
        model_name = model.name + \
            '_e' + str(epochs) + \
            '_lr_'+str(lr) + \
            '_ufratio_'+str(ratio) + \
            '_i'+str(IMG_SIZE)

        # if weights:
        #     # load weights from 'best_model.hdf5'
        #     model.load_weights(weights)

        scores = model.evaluate(val_generator, verbose=1)
        print(f"{'-'*40}\n" +
              f"before {step} training\n" +
              f"Accuracy: {scores[1]*100:.2f}\n" +
              f"{'-'*40}\n" +
              f"Start training. Unfreeze: {ratio}, LR: {lr}\n" +
              f"{'-'*40}")

        history = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks_list
        )

        scores = model.evaluate(val_generator, verbose=1)
        print(f"{'-'*21}\n" +
              f"Accuracy step {step}: {scores[1]*100:.2f}\n" +
              f"{'-'*21}")
        plot_history(history,
                     model_name,
                     f'Accuracy step {step}: {scores[1]*100:.2f}')
        # weights = 'best_model.hdf5'
        print(f'Running time: {datetime.now() - start_time}')

    if IS_ENV_COLAB:
        subprocess.run(['cp', 'best_model.hdf5',
                        '/content/drive/MyDrive/Colab Notebooks/kaggle/models'], capture_output=True)

        subprocess.run(['cp'] + glob.glob('*.png') +
                        ['/content/drive/MyDrive/Colab Notebooks/kaggle/models'], capture_output=True)

    return model


def fit(model_fun, input_shape, step, weights=None):
    # step - (lr, unfreeze_ratio, epochs)

    model, base_model = model_fun(input_shape)

    (lr, ratio, epochs) = step

    unfreeze_model(base_model, ratio)
    compile_model(model, lr)
    model_name = model.name + \
        '_e' + str(epochs) + \
        '_lr_'+str(lr) + \
        '_unfratio_'+str(ratio) + \
        '_i'+str(IMG_SIZE)

    if weights:
        # load weights from 'best_model.hdf5'
        model.load_weights(weights)

    scores = model.evaluate(val_generator, verbose=1)
    print(f"{'-'*40}\n" +
            f"before {step} training\n" +
            f"Accuracy: {scores[1]*100:.2f}\n" +
            f"{'-'*40}\n" +
            f"Start training. Unfreeze: {ratio}, LR: {lr}\n" +
            f"{'-'*40}")

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=callbacks_list
    )

    scores = model.evaluate(val_generator, verbose=1)
    print(f"{'-'*21}\n" +
            f"Accuracy step {step}: {scores[1]*100:.2f}\n" +
            f"{'-'*21}")
    plot_history(history,
                    model_name,
                    f'Accuracy step {step}: {scores[1]*100:.2f}')
    weights = 'best_model.hdf5'

    if IS_ENV_COLAB:
        subprocess.run(['cp', 'best_model.hdf5',
                        '/content/drive/MyDrive/Colab Notebooks/kaggle/models'], capture_output=True)

        subprocess.run(['cp'] + glob.glob('*.png') +
                        ['/content/drive/MyDrive/Colab Notebooks/kaggle/models'], capture_output=True)

    return model


"""
Main part
"""
# def main():
start_time = datetime.now()

K.clear_session()

train_df, sample_submission = load_data()
# plot_images()
plot_im_size(train_df)
# images, labels, images_sub = load_data()
# print(images.shape)

train_datagen = ImageDataGenerator(  # rescale=1. / 255,
    rotation_range=50,
    shear_range=0.2,
    zoom_range=[0.75, 1.25],
    brightness_range=[0.5, 1.5],
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=False,
    validation_split=VAL_SPLIT)
test_datagen = ImageDataGenerator(  # rescale=1. / 255
)

train_generator = train_datagen.flow_from_directory(
    PATH+'train/',      # директория где расположены папки с картинками
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    seed=RANDOM_SEED,
    subset='training')  # set as training data

val_generator = train_datagen.flow_from_directory(
    PATH+'train/',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True, seed=RANDOM_SEED,
    subset='validation')  # set as validation data

sub_generator = test_datagen.flow_from_dataframe(
    dataframe=sample_submission,
    directory=PATH+'test_upload/',
    x_col="Id",
    y_col=None,
    shuffle=False,
    class_mode=None,
    seed=RANDOM_SEED,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,)

# plot_images_from_generator(train_generator)
# plot_images_from_generator(val_generator)

checkpoint = ModelCheckpoint('best_model.hdf5', monitor=[
                             'val_accuracy'], verbose=1, mode='max')
earlystop = EarlyStopping(monitor='val_accuracy',
                          patience=5, restore_best_weights=True)
callbacks_list = [checkpoint, earlystop]

# work_model = models_xception_base
work_model = models_efn_base
# step - (lr, unfreeze_ratio, epochs)
steps = [(1e-3, 0, 10),
         (1e-4, 0, 10),
         (1e-4, 0.25, 5),
         (1e-4, 0.5, 5),
         (1e-5, 1, 5)]

model = fine_tune_fit(work_model, input_shape, steps)
predict_submission(model, generator=sub_generator, name=model.name)
model.save('model_224px.hdf5')
print(f'Running time: {datetime.now() - start_time}')

# single step with hi image rezolution
IMG_SIZE             = 512 # какого размера подаем изображения в сеть
IMG_CHANNELS         = 3   # у RGB 3 канала
input_shape          = (IMG_SIZE, IMG_SIZE, IMG_CHANNELS)

# step=steps[-1]
step = (1e-5, 1, 15)
model = fit(work_model, input_shape, step, weights='best_model.hdf5')
# model = fit(work_model, input_shape, step)
# input_shape, step, weights
# 'best_model.hdf5'

# single step

# model = model_xcept(lr=LR)
# model = model_xcept_new(lr=LR, unfreeze_ratio=0)
# model_name = model.name+'dl2_d025_e5' + \
#     '_lr_'+str(LR) + \
#     '_i'+str(IMG_SIZE)


# model = model_efn(lr=LR)
# model = model_efn_bnorm(lr=LR)
# model.load_weights('best_model.hdf5')
# model_name = 'efnbnorm dl2 d0 tr100 +5' + \
#     '_lr_'+str(LR) + \
#     ' i'+str(IMG_SIZE)

# # model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(lr=LR), metrics=["accuracy"])

# scores = model.evaluate(val_generator, verbose=1)
# print(f"{'-'*15}\nbefore training\nAccuracy: {scores[1]*100:.2f}\n{'-'*15}")

# history = model.fit(
#     train_generator,
#     validation_data=val_generator,
#     # steps_per_epoch = train_generator.samples//train_generator.batch_size,
#     # validation_data = test_generator,
#     # validation_steps = test_generator.samples//test_generator.batch_size,
#     epochs=EPOCHS,
#     callbacks=callbacks_list
# )

# scores = model.evaluate(val_generator, verbose=1)
# print(f"{'-'*15}\nAccuracy: {scores[1]*100:.2f}\n{'-'*15}")
# plot_history(history, model_name, f'Accuracy: {scores[1]*100:.2f}')

# subprocess.run(['cp', 'best_model.hdf5',
#                '/content/drive/MyDrive/Colab Notebooks/kaggle/models'], capture_output=True)

# subprocess.run(['cp'] + glob.glob('*.png') +
#                ['/content/drive/MyDrive/Colab Notebooks/kaggle/models'], capture_output=True)

# subprocess.run(['cp',
#                '/content/drive/MyDrive/Colab Notebooks/kaggle/models/best_model.hdf5',
#                '.'], capture_output=True)

# save current model
model.save('model_last.hdf5')
# subprocess.run(['cp', 'model_last.hdf5',
#                '/content/drive/MyDrive/Colab Notebooks/kaggle/models'], capture_output=True)

# predict

# model = work_model()
# model.load_weights('best_model.hdf5')
predict_submission(model, generator=sub_generator, name=model.name)

print(f'Running time: {datetime.now() - start_time}')

# if __name__ == '__main__':
#     main()
