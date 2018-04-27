import os
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Input, Activation, Dropout, Flatten, Dense
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import numpy as np
import glob 
from keras.preprocessing.image import img_to_array, load_img, list_pictures
from sklearn.utils import shuffle
from keras.utils import np_utils
import matplotlib.pyplot as plt
from PIL import Image
from keras.layers.normalization import BatchNormalization

classes = ['Tulip', 'Snowdrop', 'LilyValley', 'Bluebell', 'Crocus',
           'Iris', 'Tigerlily', 'Daffodil', 'Fritillary', 'Sunflower',
           'Daisy', 'ColtsFoot', 'Dandelion', 'Cowslip', 'Buttercup',
           'Windflower', 'Pansy']


result_dir = 'dataset/results'
if not os.path.exists(result_dir):
    os.mkdir(result_dir)


if __name__ == '__main__':
    row = 150
    col = 150
    channel = 3
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    class_name = []
    img_folders = glob.glob(os.path.abspath("./dataset/train_images/*"))
    for i,folder in enumerate(img_folders):
        #img_list = glob.glob(folder+"/extended/*.jpg")
        img_list = glob.glob(folder+"/*.jpg")
        class_name.append(folder)
        for img_path in img_list:
            img = img_to_array(load_img(img_path, target_size=(row, col)))
            X_train.append(img)
            Y_train.append(i)
    X_train = np.asarray(X_train)/255
    Y_train = np.asarray(Y_train)
    nb_classes = len(img_folders)
    Y_train = np_utils.to_categorical(Y_train,nb_classes)
    X_train,Y_train = shuffle(X_train,Y_train)

    img_folders = glob.glob(os.path.abspath("./dataset/test_images/*"))
    for i,folder in enumerate(img_folders):
        #img_list = glob.glob(folder+"/extended/*.jpg")
        img_list = glob.glob(folder+"/*.jpg")
        for img_path in img_list:
            img = img_to_array(load_img(img_path, target_size=(row, col)))
            X_test.append(img)
            Y_test.append(i)
    X_test = np.asarray(X_test)/255
    Y_test = np.asarray(Y_test)
    Y_test = np_utils.to_categorical(Y_test,nb_classes)
    X_test,Y_test = shuffle(X_test,Y_test)


    model = Sequential()

    model.add(Conv2D(64,3,input_shape=(row,col,channel)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPool2D(pool_size=(3,3),strides=(2,2)))

    model.add(Conv2D(128,3,padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPool2D(pool_size=(3,3),strides=(2,2)))

    model.add(Conv2D(256,3,padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPool2D(pool_size=(3,3),strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(2048))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.5))

    model.add(Dense(2048))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.5))

    model.add(Dense(nb_classes))
    model.add(Activation("softmax"))

    optimizer = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999)
    model.compile(loss="categorical_crossentropy",optimizer=optimizer, metrics=["accuracy"])

    model.summary()


    epochs = 50
    batch_size = 100
    
    hist = model.fit(X_train,Y_train,epochs=epochs,batch_size=batch_size,validation_split=0.1,verbose=1)

    score = model.evaluate(X_test,Y_test, verbose=0)
    print('test loss:', score[0])
    print('test acc:', score[1])


    loss = hist.history["loss"]
    val_loss = hist.history["val_loss"]
    acc = hist.history["acc"]
    val_acc = hist.history["val_acc"]

    nb_epochs = len(loss)
    plt.plot(range(nb_epochs), loss)
    plt.plot(range(nb_epochs), val_loss)
    plt.show()

    nb_epochs = len(acc)
    plt.plot(range(nb_epochs), acc)
    plt.plot(range(nb_epochs), val_acc)
    plt.show()    

    
    model.save_weights(os.path.join(result_dir, 'normal.h5'))
