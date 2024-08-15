import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, Concatenate, Input, BatchNormalization, Activation
from tensorflow.keras.models import Model

def UNetVGG16(input_shape=(128, 128, 3)):
    # Load the VGG16 model with batch normalization, excluding the top layers
    vgg16 = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
    
    # Extract specific layers to use in the decoder part of UNet
    encoder_layers = [
        'block1_conv2',  # 64x64x64
        'block2_conv2',  # 32x32x128
        'block3_conv3',  # 16x16x256
        'block4_conv3',  # 8x8x512
        'block5_conv3'   # 4x4x512
    ]
    
    encoder_outputs = [vgg16.get_layer(name).output for name in encoder_layers]
    
    # Create the encoder model
    encoder = Model(inputs=vgg16.input, outputs=encoder_outputs)
    
    # Input layer for UNet
    inputs = Input(shape=input_shape)
    encoder_outputs = encoder(inputs)
    
    # Decoder part of UNet
    # Starting from the bottleneck
    x = encoder_outputs[-1]
    
    # Upsampling and concatenating with corresponding encoder layers
    for i in range(len(encoder_layers) - 2, -1, -1):
        x = Conv2DTranspose(filters=512 if i >= 3 else 256 if i == 2 else 128 if i == 1 else 64, 
                            kernel_size=(2, 2), 
                            strides=(2, 2), 
                            padding='same')(x)
        x = Concatenate()([x, encoder_outputs[i]])
        x = Conv2D(filters=512 if i >= 3 else 256 if i == 2 else 128 if i == 1 else 64, 
                   kernel_size=(3, 3), 
                   padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters=512 if i >= 3 else 256 if i == 2 else 128 if i == 1 else 64, 
                   kernel_size=(3, 3), 
                   padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    
    # Output layer
    outputs = Dense(4,
                  activation='softmax',
                  kernel_initializer='he_normal')(x)
    #outputs = Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid')(x)
    
    # Create the full model
    model = Model(inputs, outputs)
    
    return model

