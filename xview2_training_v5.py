from __future__ import print_function
import numpy as np
import pandas as pd
import os
import math
import random
import argparse
import logging
import keras
from keras import backend as K
import tensorflow
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Add, Input, Concatenate
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler 
from keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping
from callbacks.epochcheckpoint import EpochCheckpoint
from callbacks.trainingmonitor import TrainingMonitor
#from callbacks.gpuCallback import GPUStats
#from keras.applications.resnet50 import ResNet50
from sklearn.utils.class_weight import compute_class_weight
from model.resnet import ResNet50v1,ResNet50v1,ResNet101v2,ordinal_loss
from model.vgg import VGG16_1,VGG16_BN,VGG16_BN1
from sklearn.metrics import f1_score
from keras.utils import CustomObjectScope


logging.basicConfig(level=logging.INFO)

# Configurations
RANDOM_SEED = 123
NUM_CLASSES = 4
BATCH_SIZE = 64
NUM_WORKERS = 4
NUM_EPOCHS = 100 

damage_intensity_encoding = dict()
damage_intensity_encoding[3] = '3'
damage_intensity_encoding[2] = '2' 
damage_intensity_encoding[1] = '1' 
damage_intensity_encoding[0] = '0'

#################################################
# Applies random transformations to training data
#################################################

###
# Creates data generator for validation set and training set
###
def validation_generator(test_csv, test_dir):
    df = pd.read_csv(test_csv)
    #df = df[:20] # delete it , for testing only
    df = df.replace({"labels" : damage_intensity_encoding })

    gen = tensorflow.keras.preprocessing.image.ImageDataGenerator(
                             rescale=1/255.0)

    #todo rescale = 1./255 ? ; original rescale = 1.4
    return gen.flow_from_dataframe(dataframe=df,
                                   directory=test_dir,
                                   x_col='uuid',
                                   y_col='labels',
                                   batch_size=BATCH_SIZE,
                                   shuffle=True,
                                   seed=RANDOM_SEED,
                                   class_mode="categorical",
                                   target_size=(128, 128)
                                   )

def augment_data(df, in_dir):

    
    df = df.replace({"labels" : damage_intensity_encoding })
    gen = tensorflow.keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True,
                             vertical_flip=True,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             rescale=1/255.0)
    print ('[INFO] performing training data augmentation')    
    #todo rescale = 1./255 ? ; original rescale = 1.4
    return gen.flow_from_dataframe(dataframe=df,
                                   directory=in_dir,
                                   x_col='uuid',
                                   y_col='labels',
                                   batch_size=BATCH_SIZE,
                                   seed=RANDOM_SEED,
                                   class_mode="categorical",
                                   target_size=(128, 128)
                                   )

#################################################
# Callbacks
#################################################
###
# Function to compute unweighted f1 scores
###
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision


    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate

    ## Note : Turn off this call back if loading a saved model since it would start
    from epoch 0 always
    """
    #lr = 1e-3
    lr = 0.1
    
    if epoch > 180:
        lr = 0.00001
    elif epoch > 120:
        lr = 0.000095
    elif epoch > 80:
        lr = 0.0005
    elif epoch > 40:
        lr = 0.00095
    elif epoch > 20:
        lr = 0.001
    elif epoch >10:
        lr = 0.01
    print('Learning rate: ', lr)
    return lr

def callbacks_func(model_out,start_epoch,plotPath,jsonPath):

    callbacks = [
            EpochCheckpoint(model_out, every=5,
                    startAt=start_epoch),
            #TrainingMonitor(plotPath,
                    #jsonPath=jsonPath,
                    #startAt=start_epoch),
            #EarlyStopping(monitor='val_loss', 
                          #patience=5,
                          #restore_best_weights=True)
            #ReduceLROnPlateau(monitor = "val_loss",
            #             factor=np.sqrt(0.1),
             #            cooldown=2,
              #           patience=5,
               #          epsilon = 1e-04,
                #         min_lr=0.5e-6,
                 #        verbose = 1),
            #LearningRateScheduler(lr_schedule)
            ]
    return callbacks


#####################################################
####################################################
# Run training and evaluation based on existing or new model
def train_model(train_data, train_csv, test_data, test_csv, model_in,callbacks):

    
    # todo : remove loading of resnet if you want saved model
    #model = ResNet101v2()
    model = VGG16_1()#UNetVGG16()
    model = VGG16_BN()
    # Add model weights if provided by user
    if model_in is not None:
        print("here...loading model")
        model.load_weights(model_in)

    df = pd.read_csv(train_csv)
    #df = df[:200]     # delete it , for testing only
    df1 = pd.read_csv(test_csv)
    #df1 = df1[:20]    # delete it , for testing only
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(df['labels'].to_list()), y=df['labels'].to_list())
    d_class_weights = dict(enumerate(class_weights))

    samples = df['uuid'].count()
    samples1 = df1['uuid'].count()
    print ('[INFO] total training images :',samples)
    print ('[INFO] total test images :',samples1)
    steps = np.ceil(samples/BATCH_SIZE)
    val_steps = np.ceil(samples1/BATCH_SIZE)

    print ('[INFO] printing class weights :',d_class_weights)

    # Augments the training data and validation data
    train_gen_flow = augment_data(df, train_data)
    validation_gen = validation_generator(test_csv, test_data)

    #Adds adam optimizer
    adam = keras.optimizers.Adam(#lr=lr_schedule(0),
                                    lr=0.001,
                                    beta_1=0.9,
                                    beta_2=0.999,
                                    decay=0.0,
                                    amsgrad=False)
    
    if model_in is None:
        print("[INFO] compiling model with cusutom loss function...")
        model.compile(loss=ordinal_loss, optimizer=adam, metrics=['accuracy', f1])

    # otherwise, we're using a checkpoint model
    else:
        print("[INFO] loading {}...".format(model_in))
        with CustomObjectScope({'ordinal_loss': ordinal_loss,'f1':f1}):
            model = load_model(model_in)
        
    
    # train the network
    print("[INFO] training network...")

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

     # Load the model and continue training
    if model_in is not None:
        with CustomObjectScope({'ordinal_loss': ordinal_loss,'f1':f1}):
            print("customobject load model")
            model = keras.models.load_model(model_in)
        #history = model.fit(X_train, y_train, epochs=additional_epochs, validation_data=(X_val, y_val))
    
    model.fit(
        train_gen_flow,
        epochs=NUM_EPOCHS,
        #initial_epoch=90,
        workers=NUM_WORKERS,
        use_multiprocessing=False,
        class_weight=d_class_weights,
        validation_data=validation_gen,
        verbose=1,
        callbacks=callbacks#.append(early_stopping)
    )

    model.save_weights('ResNet101v2_weights.h5')
    #Evalulate f1 weighted scores on validation set
    predictions = model.predict(validation_gen)

    val_trues = validation_gen.classes
    val_pred = np.argmax(predictions, axis=-1)

    try:
        f1_weighted = f1_score(val_trues, val_pred, average='weighted')
        print ('F1 weighted score calculated : ',f1_weighted)
    except:
        f1_weighted = f1_score(val_trues, val_pred,average='weighted')
        print ('F1 score calculated : ',f1_weighted)

def main():

    parser = argparse.ArgumentParser(description='Run Building Damage Classification Training & Evaluation')
    parser.add_argument('--train_data',
                        required=True,
                        metavar="/path/to/xBD_train",
                        help="Full path to the train data directory")
    parser.add_argument('--train_csv',
                        required=True,
                        metavar="/path/to/xBD_split",
                        help="Full path to the train csv")
    parser.add_argument('--test_data',
                        required=True,
                        metavar="/path/to/xBD_test",
                        help="Full path to the test data directory")
    parser.add_argument('--test_csv',
                        required=True,
                        metavar="/path/to/xBD_split",
                        help="Full path to the test csv")
    parser.add_argument('--model_in',
                        required = False,
                        default=None,
                        metavar='/path/to/input_model',
                        help="Path to saved model")
    parser.add_argument('--model_out',
                        required=True,
                        metavar='/path/to/save_model',
                        help="Path to Output directory")
    parser.add_argument('--start_epoch',
                        required=False,
                        type = int,
                        default = 0,
                        metavar='30',
                        help="epoch to restart training at")

    args = parser.parse_args()

    logging.info("Started Model Training")

    csv_path = args.train_csv
    train_csv_path = csv_path+'/'+'train.csv'

    csv_path1 = args.test_csv
    test_csv_path = csv_path1+'/'+'test.csv'

    # build the path to the training plot and training history

    plotPath = os.path.sep.join([args.model_out, "xview2_train.png"])
    jsonPath = os.path.sep.join([args.model_out, "xviw2_train.json"])
    print("[INFO] saving to {}...".format(args.model_out))
    callbacks = callbacks_func(args.model_out,args.start_epoch,plotPath,jsonPath)
    train_model(args.train_data,train_csv_path,args.test_data,test_csv_path,args.model_in,callbacks)   
    logging.info("Successfully Completed Training")
    
if __name__ == '__main__':
    main()
