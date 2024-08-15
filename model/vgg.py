import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten,Input, BatchNormalization, Dropout,Conv2D, MaxPooling2D, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


def VGG16_1():
    # Load the VGG16 model with batch normalization, excluding the top (fully connected) layers
    inputs = Input(shape=(128, 128, 3))
    base_model = VGG16(include_top=False, weights='imagenet', input_shape=(128, 128, 3))

    for layer in base_model.layers:
        layer.trainable = False

    base_vgg = base_model(inputs)
    base_vgg = Flatten()(base_vgg)

    output = Dense(4,
                    activation='softmax',
                    kernel_initializer='he_normal')(base_vgg)
    
    model = Model(inputs=inputs, outputs=output)
    print("[INFO] Extracting VGG model...")
    return model

def VGG16_BN():
    base_model = VGG16(include_top=False, weights='imagenet', input_shape=(128, 128, 3))

    # Freeze the base model
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom layers on top of the base model
    x = base_model.output
    x = Flatten()(x)
    x = Dense(2048, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    outputs = Dense(4, activation='softmax',kernel_initializer='he_normal')(x) 

    # Create the new model
    model = Model(inputs=base_model.input, outputs=outputs)
    print("[INFO] Extracting VGG16_BN model...")
    return model

def VGG16_BN1():
  weights = 'imagenet'
  inputs = Input(shape=(128, 128, 3))

  base_model = VGG16(include_top=False, weights=weights, input_shape=(128, 128, 3))

  for layer in base_model.layers:
    layer.trainable = False

  x = Conv2D(32, (5, 5), strides=(1, 1), padding='same', activation='relu', input_shape=(128, 128, 3))(inputs)
  x = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(x)

  x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
  x = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(x)

  x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
  x = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(x)

  x = Flatten()(x)

  base_vgg = base_model(inputs)
  base_vgg = Flatten()(base_vgg)

  concated_layers = Concatenate()([x, base_vgg])

  concated_layers = Dense(2024, activation='relu')(concated_layers)
  concated_layers = BatchNormalization()(concated_layers)
  concated_layers = Dropout(0.5)(concated_layers)
  concated_layers = Dense(524, activation='relu')(concated_layers)
  concated_layers = BatchNormalization()(concated_layers)
  concated_layers = Dropout(0.5)(concated_layers)
  concated_layers = Dense(124, activation='relu')(concated_layers)
  output = Dense(4, activation='relu')(concated_layers)

  model = Model(inputs=inputs, outputs=output)
  print("[INFO] Extracting VGG16_BN1 model...")
  return model
