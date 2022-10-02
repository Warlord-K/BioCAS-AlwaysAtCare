import tensorflow as tf
import numpy as np
import os
tf.get_logger().setLevel('ERROR')

class Model:
    def __init__(self):
        self.create_models()


    def create_models(self):
        IMG_SHAPE = (224, 224, 3)

        conv_layer = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                    include_top=False,
                                                    weights='imagenet')
        conv_layer.trainable = False

        pooling_layer = tf.keras.layers.GlobalAveragePooling2D()
        hidden_layer1 = tf.keras.layers.Dense(32,activation = "relu")
        hidden_layer2 = tf.keras.layers.Dense(32,activation = "relu")
        output_layer_binary = tf.keras.layers.Dense(1,activation="softmax")
        output_layer_multi = tf.keras.layers.Dense(7,activation="softmax")
        output_layer_ternary = tf.keras.layers.Dense(3,activation="softmax")
        output_layer_multi_2 = tf.keras.layers.Dense(5,activation = "softmax")
        self.model_1 = tf.keras.Sequential([
        conv_layer,
        pooling_layer,
        hidden_layer1,
        hidden_layer2,
        output_layer_binary
        ])
        self.model_2 = tf.keras.Sequential([
        conv_layer,
        pooling_layer,
        hidden_layer1,
        hidden_layer2,
        output_layer_multi
        ])
        self.model_3 = tf.keras.Sequential([
        conv_layer,
        pooling_layer,
        hidden_layer1,
        hidden_layer2,
        output_layer_ternary
        ])
        self.model_4 = tf.keras.Sequential([
        conv_layer,
        pooling_layer,
        hidden_layer1,
        hidden_layer2,
        output_layer_multi_2
        ])
    

    

    def predict(self,images,mode = 1):
        img  = tf.data.Dataset.from_tensor_slices(images).batch(1)
        path = os.path.realpath(__file__)[:-10]
        if(mode == 11):
            self.model_1.load_weights(f"{path}\checkpoints\mobilenetv2_binary_100")
            pred = self.model_1.predict(img)
            predictions = []
            for i in range(len(pred)):
                pred_ = "Normal" if pred[i][0] == 0 else "Adventititous"
                predictions.append(pred_)
        elif(mode == 12):
            self.model_2.load_weights(f'{path}\checkpoints\checkpoint_90epochs')
            pred = self.model_2.predict(img)
            decoder_arr = ['Normal', 'Fine Crackle', 'Wheeze', 'Rhonchi', 'Coarse Crackle','Wheeze+Crackle', 'Stridor']
            predictions = []
            for i in range(len(pred)):
                pred_ = decoder_arr[np.argmax(pred[i])]
                predictions.append(pred_)
        elif(mode == 21):
            self.model_3.load_weights(fr'{path}\checkpoints\2_1_60epochs')
            pred = self.model_3.predict(img)
            decoder_arr = ['Normal','Adventitious','Poor Quality']
            predictions = []
            for i in range(len(pred)):
                pred_ = decoder_arr[np.argmax(pred[i])]
                predictions.append(pred_)
        elif(mode == 22):
            self.model_4.load_weights(fr'{path}\checkpoints\record_multi_70epochs')
            pred = self.model_4.predict(img)
            decoder_arr = ['CAS','CAS & DAS','DAS','Normal',"Poor Quality"]
            predictions = []
            for i in range(len(pred)):
                pred_ = decoder_arr[np.argmax(pred[i])]
                predictions.append(pred_)
        
        return predictions
    
    

    
        
