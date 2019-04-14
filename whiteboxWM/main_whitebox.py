import DeepSigns
from DeepSigns import subsample_training_data
from DeepSigns import WM_activity_regularizer
from DeepSigns import get_activations
from DeepSigns import extract_WM_from_activations 
from DeepSigns import compute_BER

from utils import create_marked_model


import numpy as np 
from keras.datasets import mnist
from keras.models import Model
from keras.models import Sequential
import keras.utils.np_utils as kutils
from keras.layers import Dense, Dropout, Input
from keras.optimizers import RMSprop 
import keras.callbacks as callbacks

## ------ Demo of white-box activation watermarking on MNIST-MLP benchmark ---- ##
if __name__ == '__main__':

    (x_train, y_train_vec), (x_test, y_test_vec) = mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    num_classes = 10 
    y_train = kutils.to_categorical(y_train_vec, num_classes)
    y_test = kutils.to_categorical(y_test_vec, num_classes)
    
    
    ## WM configs ---- ##
    scale = 0.01      # for loss1
    gamma2 = 0.01     # for loss2 
    target_dense_idx = 2      #  target layer to carry WM
    embed_bits = 16      
    target_class = 0
    epochs = 1

    b = np.random.randint(2, size=(embed_bits, num_classes))  # binary prior info to be embedded, shape (T, 10)
    aux_ip = Input(shape=[None], name='aux_input')
    WM_reg = WM_activity_regularizer(gamma1=scale, gamma2=gamma2, b=b, target_class=target_class, label=aux_ip,num_classes=num_classes)  

    ## ---- Build model ----- ##
    main_ip = Input(shape=(784, ), name='main_input')
    x = Dense(512, activation='relu', input_shape=(784,))(main_ip)
    x = Dropout(0.2)(x)
    marked_FC = Dense(512, activation='relu', activity_regularizer=WM_reg)
    x = marked_FC(x)
    marked_FC.trainable_weights=marked_FC.trainable_weights+[WM_reg.centers]
    x = Dropout(0.2)(x)
    x = Dense(num_classes, activation='softmax')(x)
    model = Model(input=[main_ip, aux_ip], output=x)

    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.001),
                  metrics=['accuracy'])         

    history = model.fit([x_train, y_train_vec], y_train, nb_epoch=epochs, verbose=1,
                        validation_data=([x_test, y_test_vec], y_test))
    score = model.evaluate([x_test, y_test_vec], y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    marked_FC.trainable_weights = marked_FC.trainable_weights[0:2]
    model.save_weights('result/wmarked_weights.h5')


    ## ---- Validate WM ---- ##
    marked_model = create_marked_model()
    marked_model.summary()
    marked_model.load_weights('result/wmarked_weights.h5')
    marked_model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

    x_train_subset, y_train_subset = subsample_training_data(x_train, y_train_vec, target_class)
    marked_activations = get_activations(marked_model, x_train_subset, print_shape_only=True)        # this is list
    print("Get activations of marked FC layer")
    marked_FC_activations = marked_activations[target_dense_idx+1]           # choose the activations from first wmarked dense layer
    A = np.load('result/projection_matrix.npy')
    print('A = ', A)
    decoded_WM = extract_WM_from_activations(marked_FC_activations, A)
    BER = compute_BER(decoded_WM, b[:, target_class])
    print("BER in class {} is {}: ".format(target_class, BER))
