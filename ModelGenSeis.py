from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D,Dropout,UpSampling2D,Input,MaxPooling2D,Conv2DTranspose,concatenate,BatchNormalization,ReLU
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2


def DownSampleLayer2D(input_layer,num_filt,kernel,start_filters=8):
    
        conv = Conv2D(start_filters * num_filt, kernel, activation="relu", padding="same")(input_layer)   
        conv = BatchNormalization()(conv)
        conv = ReLU()(conv)
        conv = Conv2D(start_filters * num_filt, kernel, activation="relu", padding="same")(conv)   
        conv = BatchNormalization()(conv)
        conv = ReLU()(conv)   
        pool = MaxPooling2D((2,2),padding='valid', strides=2)(conv) 
        
        return pool,conv

def UpSampleLayer2D(input_layer,concat_layer,num_filt,kernel,start_filters=8):
    
        deconv = Conv2DTranspose(start_filters * num_filt, kernel,strides=(2,2),padding="same")(input_layer)
        uconv = concatenate([deconv, concat_layer])
        uconv = Conv2D(start_filters * num_filt, kernel, activation="relu", padding="same")(uconv)   
        uconv = BatchNormalization()(uconv)
        uconv = ReLU()(uconv)
        uconv = Conv2D(start_filters * num_filt, kernel, activation="relu", padding="same")(uconv)   
        uconv = BatchNormalization()(uconv)
        uconv = ReLU()(uconv)      
        
        return uconv

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
    # input_shape = (128,128,1)
    input_layer = Input(shape=(input_size))

    if type=="Unet_v3":
      sf = start_filters
      d1,c1 = DownSampleLayer2D(input_layer,num_filt=1,start_filters=sf, kernel=kernel)                                                       
      d2,c2 = DownSampleLayer2D(d1,num_filt=2, start_filters=sf,kernel=kernel)   
      d3,c3 = DownSampleLayer2D(d2,num_filt=4, start_filters=sf,kernel=kernel)   
      d4,c4 = DownSampleLayer2D(d3,num_filt=8, start_filters=sf,kernel=kernel)   
      d5,c5 = DownSampleLayer2D(d4,num_filt=16, start_filters=sf,kernel=kernel)   

      convm = Conv2D(sf * 32, kernel, activation="relu", padding="same")(d5)
      convm = Conv2D(sf * 32, kernel, activation="relu", padding="same")(convm)    
      
      u5 = UpSampleLayer2D(convm,c5, start_filters=sf,num_filt=16, kernel=kernel)
      u4 = UpSampleLayer2D(u5,c4,start_filters=sf,num_filt=8, kernel=kernel)
      u3 = UpSampleLayer2D(u4,c3,start_filters=sf,num_filt=4, kernel=kernel)
      u2 = UpSampleLayer2D(u3,c2,start_filters=sf,num_filt=2, kernel=kernel)
      u1 = UpSampleLayer2D(u2,c1,start_filters=sf,num_filt=1, kernel=kernel)
      output_layer = Conv2D(1, (1,1),activation='linear', padding="same")(u1)

      model = Model(input_layer,output_layer)

        
    elif type=="Unet":
        input_shape = input_size
        input_layer = Input(shape=(input_shape))
        # # LHS of UNET
        conv1 = Conv2D(start_filters * 1, kernel, activation="relu", padding="same")(input_layer)   #128x128
        conv1 = Conv2D(start_filters * 1, kernel, activation="relu", padding="same")(conv1)
        pool1 = MaxPooling2D((2, 2))(conv1)                                                         #64x64
        pool1 = Dropout(0.5)(pool1)

        conv2 = Conv2D(start_filters * 4, kernel, activation="relu", padding="same")(pool1)
        conv2 = Conv2D(start_filters * 4, kernel, activation="relu", padding="same")(conv2)
        pool2 = MaxPooling2D((2, 2))(conv2)                                                         #32x32
        pool2 = Dropout(0.5)(pool2)

        conv3 = Conv2D(start_filters * 4, kernel, activation="relu", padding="same")(pool2)
        conv3 = Conv2D(start_filters * 4, kernel, activation="relu", padding="same")(conv3)
        pool3 = MaxPooling2D((2, 2))(conv3)                                                         #16x16
        pool3 = Dropout(0.5)(pool3)

        conv4 = Conv2D(start_filters * 8, kernel, activation="relu", padding="same")(pool2)
        conv4 = Conv2D(start_filters * 8, kernel, activation="relu", padding="same")(conv4)
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

        deconv2 = Conv2DTranspose(start_filters * 4, kernel, strides=(2, 2), padding="same")(uconv4)
        uconv2 = concatenate([deconv2, conv2])
        uconv2 = Dropout(0.5)(uconv2)
        uconv2 = Conv2D(start_filters * 4, kernel, activation="relu", padding="same")(uconv2)
        uconv2 = Conv2D(start_filters * 4, kernel, activation="relu", padding="same")(uconv2)

        deconv1 = Conv2DTranspose(start_filters * 1, kernel, strides=(2, 2), padding="same")(uconv2)
        uconv1 = concatenate([deconv1, conv1])
        uconv1 = Dropout(0.5)(uconv1)
        uconv1 = Conv2D(start_filters * 1, kernel, activation="relu", padding="same")(uconv1)
        uconv1 = Conv2D(start_filters * 1, kernel, activation="relu", padding="same")(uconv1)

        output_layer = Conv2D(1, (1,1),activation='linear', padding="same")(uconv1)

        model = Model(input_layer,output_layer)

    if type=="Unet_v2":
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

    elif type == 'upsc_v2':
        model = Sequential()
        model.add(Conv2D(filters = start_filters*2, kernel_size = kernel,padding = 'Same', 
                        activation ='relu',input_shape=input_size))
        model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
        # model.add(Dropout(0.2))
        model.add(Conv2D(filters = start_filters*4, kernel_size = kernel,padding = 'Same', 
                        activation ='relu'))
        model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
        # model.add(Dropout(0.2))

        model.add(Conv2D(filters = start_filters*8, kernel_size = kernel,padding = 'Same', 
                        activation ='relu'))
        model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
        # model.add(Dropout(0.2))

        model.add(Conv2D(filters = start_filters*16, kernel_size = kernel,padding = 'Same', 
                        activation ='relu'))
        model.add(Conv2D(filters = start_filters*16, kernel_size = kernel,padding = 'Same', 
                        activation ='relu'))

        model.add(UpSampling2D((2,2)))
        model.add(Conv2D(filters = start_filters*8, kernel_size = kernel,padding = 'Same', 
                        activation ='relu'))
        model.add(UpSampling2D((2,2)))
        model.add(Conv2D(filters = start_filters*4, kernel_size = kernel,padding = 'Same', 
                        activation ='relu'))
        model.add(UpSampling2D((2,2)))
        model.add(Conv2D(filters = start_filters*4, kernel_size = kernel,padding = 'Same', 
                        activation ='relu'))

        model.add(Conv2D(1,(3,3),activation='linear',padding='same'))

    elif type == 'upsc_v3':
        model = Sequential()
        model.add(Conv2D(filters = start_filters*2, kernel_size = kernel,padding = 'Same', 
                        activation ='relu',input_shape=input_size))
        model.add(MaxPooling2D(pool_size=(2,2),padding='same'))

        model.add(Conv2D(filters = start_filters*4, kernel_size = kernel,padding = 'Same', 
                        activation ='relu'))
        model.add(MaxPooling2D(pool_size=(2,2),padding='same'))

        model.add(Conv2D(filters = start_filters*8, kernel_size = kernel,padding = 'Same', 
                        activation ='relu'))
        model.add(MaxPooling2D(pool_size=(2,2),padding='same'))

        model.add(Conv2D(filters = start_filters*16, kernel_size = kernel,padding = 'Same', 
                        activation ='relu'))
        model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
        model.add(Dropout(0.1))

        model.add(Conv2D(filters = start_filters*16, kernel_size = kernel,padding = 'Same', 
                        activation ='relu'))
        model.add(UpSampling2D((2,2)))
        model.add(Conv2D(filters = start_filters*8, kernel_size = kernel,padding = 'Same', 
                        activation ='relu'))
        model.add(UpSampling2D((2,2)))
        model.add(Conv2D(filters = start_filters*4, kernel_size = kernel,padding = 'Same', 
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

        encoded1 = Dense(512,activation='relu')(input_layer)
        encoded2 = Dense(256,activation='relu')(encoded1)
        encoded3 = Dense(128,activation='relu')(encoded2)
        encoded4 = Dense(64,activation='relu')(encoded3)
        encoded5 = Dense(32,activation='relu')(encoded4)
        encoded6 = Dense(16,activation='relu')(encoded5)

        encoded6 = Dropout(0.2)(encoded6)

        decoded1 = Dense(16,activation='relu')(encoded6)
        decoded2 = Dense(32,activation='relu')(decoded1)
        decoded3 = Dense(64,activation='relu')(decoded2)
        decoded4 = Dense(128,activation='relu')(encoded3)
        decoded5 = Dense(256,activation='relu')(decoded4)
        decoded6 = Dense(512,activation='relu')(decoded5)

        decoded = Dense(256, activation='linear')(decoded6)


        model = Model(input_layer, decoded)

    return model