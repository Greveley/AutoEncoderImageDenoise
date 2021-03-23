from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D,Dropout,UpSampling2D,Input,MaxPooling2D,Conv2DTranspose,concatenate
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Model

def autoencoder(type, start_filters=8, kernel=(3,3),input_size=(128,128,1)):
    """
        Generates a range of Autoencoder models for 2D image denoising
        Type :  UpSc = Uses MaxPooling and Upscaling to generate a simple denoising autoencoder 
                Tran = Literally uses a Convolution and a Transpose Convolution to denoise the data
                Unet = Uses convolutional NN with the Unet architecture to generate the model (str)
        Start_filters : Number of filters in the first layer of the model, each layer increases the number of filters
                        in the model by a factor of 2 to the centre (where the upscaling begings) (int) 
        kernel :       Shape of convolutional kernel to be used in the model (tuple)
    """
    input_shape = (128,128,1)
    input_layer = Input(shape=(input_shape))

    if type=="Unet_NoSc_v2":
        input_shape = input_size
        input_layer = Input(shape=(input_shape))
        # # LHS of UNET
        conv1 = Conv2D(start_filters * 1, kernel, activation="relu", padding="same")(input_layer)  
        conv1 = Conv2D(start_filters * 1, kernel, activation="relu", padding="same")(conv1)
        pool1 = MaxPooling2D((2, 2))(conv1)                                                         
        pool1 = Dropout(0.25)(pool1)

        conv2 = Conv2D(start_filters * 2, kernel, activation="relu", padding="same")(pool1)
        conv2 = Conv2D(start_filters * 2, kernel, activation="relu", padding="same")(conv2)
        pool2 = MaxPooling2D((2, 2))(conv2)                                                        
        pool2 = Dropout(0.5)(pool2)

        conv3 = Conv2D(start_filters * 4, kernel, activation="relu", padding="same")(pool2)
        conv3 = Conv2D(start_filters * 4, kernel, activation="relu", padding="same")(conv3)
        pool3 = MaxPooling2D((2, 2))(conv3)                                                         
        pool3 = Dropout(0.5)(pool3)

        # # Middle
        convm = Conv2D(start_filters * 16, kernel, activation="relu", padding="same")(pool3)
        convm = Conv2D(start_filters * 16, kernel, activation="relu", padding="same")(convm)

        deconv3 = Conv2DTranspose(start_filters * 4, kernel, strides=(2,2), padding="same")(convm)
        uconv3 = concatenate([deconv3, conv3])
        uconv3 = Dropout(0.5)(uconv3)
        uconv3 = Conv2D(start_filters * 4, kernel, activation="relu", padding="same")(uconv3)
        uconv3 = Conv2D(start_filters * 4, kernel, activation="relu", padding="same")(uconv3)

        deconv2 = Conv2DTranspose(start_filters * 2, kernel, strides=(2, 2), padding="same")(uconv3)
        uconv2 = concatenate([deconv2, conv2])
        uconv2 = Dropout(0.5)(uconv2)
        uconv2 = Conv2D(start_filters * 2, kernel, activation="relu", padding="same")(uconv2)
        uconv2 = Conv2D(start_filters * 2, kernel, activation="relu", padding="same")(uconv2)

        deconv1 = Conv2DTranspose(start_filters * 1, kernel, strides=(2, 2), padding="same")(uconv2)
        uconv1 = concatenate([deconv1, conv1])
        uconv1 = Dropout(0.5)(uconv1)
        uconv1 = Conv2D(start_filters * 1, kernel, activation="relu", padding="same")(uconv1)
        uconv1 = Conv2D(start_filters * 1, kernel, activation="relu", padding="same")(uconv1)

        output_layer = Conv2D(1, (1,1), padding="same", activation="linear")(uconv1)

        model = Model(input_layer,output_layer)

    elif type=="Unet_NoSc":
        input_shape = input_size
        input_layer = Input(shape=(input_shape))
        # # LHS of UNET
        conv1 = Conv2D(start_filters * 1, kernel, activation="relu", padding="same")(input_layer)   #128x128
        conv1 = Conv2D(start_filters * 1, kernel, activation="relu", padding="same")(conv1)
        pool1 = MaxPooling2D((2, 2))(conv1)                                                         #64x64
        pool1 = Dropout(0.5)(pool1)

        conv2 = Conv2D(start_filters * 2, kernel, activation="relu", padding="same")(pool1)
        conv2 = Conv2D(start_filters * 2, kernel, activation="relu", padding="same")(conv2)
        pool2 = MaxPooling2D((2, 2))(conv2)                                                         #32x32
        pool2 = Dropout(0.5)(pool2)

        conv3 = Conv2D(start_filters * 4, kernel, activation="relu", padding="same")(pool2)
        conv3 = Conv2D(start_filters * 4, kernel, activation="relu", padding="same")(conv3)
        pool3 = MaxPooling2D((2, 2))(conv3)                                                         #16x16
        pool3 = Dropout(0.5)(pool3)

        conv4 = Conv2D(start_filters * 4, kernel, activation="relu", padding="same")(pool3)
        conv4 = Conv2D(start_filters * 4, kernel, activation="relu", padding="same")(conv4)
        pool4 = MaxPooling2D((2, 2))(conv4)                                                         #8x8
        pool4 = Dropout(0.5)(pool4)

        # # Middle
        convm = Conv2D(start_filters * 16, kernel, activation="relu", padding="same")(pool4)
        convm = Conv2D(start_filters * 16, kernel, activation="relu", padding="same")(convm)

        # # RHS of UNET
        deconv4 = Conv2DTranspose(start_filters * 8, kernel,strides=(2,2),padding="same")(convm)
        uconv4 = concatenate([deconv4, conv4])
        uconv4 = Dropout(0.5)(uconv4)
        uconv4 = Conv2D(start_filters * 8, kernel, activation="relu", padding="same")(uconv4)
        uconv4 = Conv2D(start_filters * 8, kernel, activation="relu", padding="same")(uconv4)

        deconv3 = Conv2DTranspose(start_filters * 4, kernel, strides=(2,2), padding="same")(uconv4)
        uconv3 = concatenate([deconv3, conv3])
        uconv3 = Dropout(0.5)(uconv3)
        uconv3 = Conv2D(start_filters * 4, kernel, activation="relu", padding="same")(uconv3)
        uconv3 = Conv2D(start_filters * 4, kernel, activation="relu", padding="same")(uconv3)

        deconv2 = Conv2DTranspose(start_filters * 2, kernel, strides=(2, 2), padding="same")(uconv3)
        uconv2 = concatenate([deconv2, conv2])
        uconv2 = Dropout(0.5)(uconv2)
        uconv2 = Conv2D(start_filters * 2, kernel, activation="relu", padding="same")(uconv2)
        uconv2 = Conv2D(start_filters * 2, kernel, activation="relu", padding="same")(uconv2)

        deconv1 = Conv2DTranspose(start_filters * 1, kernel, strides=(2, 2), padding="same")(uconv2)
        uconv1 = concatenate([deconv1, conv1])
        uconv1 = Dropout(0.5)(uconv1)
        uconv1 = Conv2D(start_filters * 1, kernel, activation="relu", padding="same")(uconv1)
        uconv1 = Conv2D(start_filters * 1, kernel, activation="relu", padding="same")(uconv1)

        output_layer = Conv2D(1, (1,1),activation='linear', padding="same")(uconv1)

        model = Model(input_layer,output_layer)

    if type=="Unet_NoSc_v3":
        input_shape = input_size
        input_layer = Input(shape=(input_shape))
        # # LHS of UNET
        conv1 = Conv2D(start_filters * 1, kernel, activation="relu", padding="same")(input_layer)  
        conv1 = Conv2D(start_filters * 1, kernel, activation="relu", padding="same")(conv1)
        pool1 = MaxPooling2D((2, 2))(conv1)                                                         
        pool1 = Dropout(0.25)(pool1)

        conv2 = Conv2D(start_filters * 4, kernel, activation="relu", padding="same")(pool1)
        conv2 = Conv2D(start_filters * 4, kernel, activation="relu", padding="same")(conv2)
        pool2 = MaxPooling2D((2, 2))(conv2)                                                        
        pool2 = Dropout(0.5)(pool2)

        # # Middle
        convm = Conv2D(start_filters * 16, kernel, activation="relu", padding="same")(pool2)
        convm = Conv2D(start_filters * 16, kernel, activation="relu", padding="same")(convm)

        deconv2 = Conv2DTranspose(start_filters * 4, kernel, strides=(2, 2), padding="same")(convm)
        uconv2 = concatenate([deconv2, conv2])
        uconv2 = Dropout(0.5)(uconv2)
        uconv2 = Conv2D(start_filters * 4, kernel, activation="relu", padding="same")(uconv2)
        uconv2 = Conv2D(start_filters * 4, kernel, activation="relu", padding="same")(uconv2)

        deconv1 = Conv2DTranspose(start_filters * 1, kernel, strides=(2, 2), padding="same")(uconv2)
        uconv1 = concatenate([deconv1, conv1])
        uconv1 = Dropout(0.5)(uconv1)
        uconv1 = Conv2D(start_filters * 1, kernel, activation="relu", padding="same")(uconv1)
        uconv1 = Conv2D(start_filters * 1, kernel, activation="relu", padding="same")(uconv1)

        output_layer = Conv2D(1, (1,1), padding="same", activation="linear")(uconv1)

        model = Model(input_layer,output_layer)


    elif type == 'upsc':
        model = Sequential()
        model.add(Conv2D(filters = start_filters*1, kernel_size = kernel,padding = 'Same', 
                        activation ='relu',input_shape=input_size))
        model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
        model.add(Conv2D(filters = start_filters*4, kernel_size = kernel,padding = 'Same', 
                        activation ='relu'))
        model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
        model.add(Conv2D(filters = start_filters*8, kernel_size = kernel,padding = 'Same', 
                        activation ='relu'))
        model.add(MaxPooling2D(pool_size=(2,2),padding='same'))

        model.add(UpSampling2D((2,2)))
        model.add(Conv2D(filters = start_filters*8, kernel_size = kernel,padding = 'Same', 
                        activation ='relu'))
        model.add(UpSampling2D((2,2)))
        model.add(Conv2D(filters = start_filters*4, kernel_size = kernel,padding = 'Same', 
                        activation ='relu'))
        model.add(UpSampling2D((2,2)))
        model.add(Conv2D(filters = start_filters*1, kernel_size = kernel,padding = 'Same', 
                        activation ='relu'))
        model.add(Conv2D(1,(3,3),activation='linear',padding='same'))

    elif type == 'upsc_v2':
        model = Sequential()
        model.add(Conv2D(filters = start_filters*2, kernel_size = kernel,padding = 'Same', 
                        activation ='relu',input_shape=input_size))
        model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
        model.add(Conv2D(filters = start_filters*8, kernel_size = kernel,padding = 'Same', 
                        activation ='relu'))
        model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
        model.add(Conv2D(filters = start_filters*8, kernel_size = kernel,padding = 'Same', 
                        activation ='relu'))
        model.add(MaxPooling2D(pool_size=(2,2),padding='same'))

        model.add(Conv2D(filters = start_filters*8, kernel_size = kernel,padding = 'Same', 
                        activation ='relu'))
        model.add(UpSampling2D((2,2)))
        model.add(Conv2D(filters = start_filters*8, kernel_size = kernel,padding = 'Same', 
                        activation ='relu'))
        model.add(UpSampling2D((2,2)))
        model.add(Conv2D(filters = start_filters*2, kernel_size = kernel,padding = 'Same', 
                        activation ='relu'))
        model.add(UpSampling2D((2,2)))
        model.add(Conv2D(1,(3,3),activation='linear',padding='same'))


    elif type == 'upsc_v2_dr':
        model = Sequential()
        model.add(Conv2D(filters = start_filters*2, kernel_size = kernel,padding = 'Same', 
                        activation ='relu',input_shape=input_size))
        model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
        model.add(Dropout(0.5))
        model.add(Conv2D(filters = start_filters*8, kernel_size = kernel,padding = 'Same', 
                        activation ='relu'))
        model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
        model.add(Dropout(0.5))
        model.add(Conv2D(filters = start_filters*8, kernel_size = kernel,padding = 'Same', 
                        activation ='relu'))
        model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
        # model.add(Dropout(0.5))

        model.add(Conv2D(filters = start_filters*8, kernel_size = kernel,padding = 'Same', 
                        activation ='relu'))
        model.add(UpSampling2D((2,2)))
        model.add(Conv2D(filters = start_filters*8, kernel_size = kernel,padding = 'Same', 
                        activation ='relu'))
        model.add(UpSampling2D((2,2)))
        model.add(Conv2D(filters = start_filters*2, kernel_size = kernel,padding = 'Same', 
                        activation ='relu'))
        model.add(UpSampling2D((2,2)))
        model.add(Conv2D(1,(3,3),activation='linear',padding='same'))


    elif type == 'ANN':

        # input_img = Input(shape=(INPUT_SIZE2,))

        input_shape = (256,)
        input_layer = Input(shape=(input_shape))

        encoded1 = Dense(512, activation='relu')(input_layer)
        # encoded1 = Dropout(0.5)(encoded1)
        encoded2 = Dense(256, activation='relu')(encoded1)
        # encoded2 = Dropout(0.5)(encoded2)
        encoded3 = Dense(128, activation='relu')(encoded2)
        # encoded4 = Dense(64, activation='relu')(encoded3)
        # encoded3 = Dropout(0.5)(encoded3)
        # decoded1 = Dense(64, activation='relu')(encoded4)
        decoded2 = Dense(128, activation='relu')(encoded3)
        # decoded1 = Dropout(0.3)(decoded1)
        decoded3 = Dense(256, activation='relu')(decoded2)
        # decoded2 = Dropout(0.5)(decoded2)
        decoded4 = Dense(512, activation='relu')(decoded3)
        # decoded3 = Dropout(0.5)(decoded3)

        decoded = Dense(256, activation='linear')(decoded4)


        model = Model(input_layer, decoded)

    elif type == 'ANN_dr':

        # input_img = Input(shape=(INPUT_SIZE2,))

        input_shape = (256,)
        input_layer = Input(shape=(input_shape))

        encoded1 = Dense(512, activation='relu')(input_layer)
        # encoded1 = Dropout(0.2)(encoded1)
        encoded2 = Dense(256, activation='relu')(encoded1)
        # encoded2 = Dropout(0.2)(encoded2)
        encoded3 = Dense(128, activation='relu')(encoded2)
        encoded3 = Dropout(0.2)(encoded3)

        decoded1 = Dense(128, activation='relu')(encoded3)
        decoded1 = Dropout(0.2)(decoded1)
        decoded2 = Dense(256, activation='relu')(decoded1)
        # decoded2 = Dropout(0.2)(decoded2)
        decoded3 = Dense(512, activation='relu')(decoded2)
        # decoded3 = Dropout(0.2)(decoded3)

        decoded = Dense(256, activation='linear')(decoded3)


        model = Model(input_layer, decoded)

    return model