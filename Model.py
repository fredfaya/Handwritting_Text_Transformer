import tensorflow as tf
from tensorflow import keras
import os
import wandb
from wandb.keras import WandbCallback

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class CTCLayer(keras.layers.Layer):
    def __int__(self, nane=None):
        super().__init__(name=nane)
        # self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        return y_pred


class Model:

    def __init__(self,
                 input_shape,
                 datapreprocessor
                 ):
        self.RNNOutput = None
        self.CNNOutput = None
        self.input_img = keras.Input(shape=(input_shape[0], input_shape[1], 1), name="image")
        self.labels = keras.layers.Input(name="label", shape=(None,))
        self.input_shape = input_shape
        self.datapreprocessor = datapreprocessor

        # creattion des layers de notre model
        self.setupCNN()
        self.setupRNN()
        self.setupCTC()

    def setupCNN(self):
        # liste pour avoir les parametres des layers
        kernel_vals = [5, 5, 3, 3, 3]
        feature_vals = [1, 32, 64, 128, 128, 256]
        stride_vals = pool_vals = [(2, 2), (2, 2), (2, 2), (2, 2), (2, 2)]
        num_layers = len(stride_vals)

        pool = self.input_img
        for i in range(num_layers):
            conv = keras.layers.Conv2D(kernel_size=(kernel_vals[i], kernel_vals[i]), filters=64, activation="relu",
                                       padding='SAME',
                                       strides=(1, 1),
                                       name='conv_layer' + str(i))(pool)
            conv_norm = keras.layers.BatchNormalization()(conv)  # couche de normalization
            pool = keras.layers.MaxPool2D(pool_size=pool_vals[i],
                                          strides=stride_vals[i][0],
                                          padding='VALID')(conv_norm)  # couche de pooling

        new_shape = ((self.input_shape[0] // 4), (self.input_shape[1] // 4))
        pool = keras.layers.Reshape(target_shape=new_shape, name="reshape")(pool)
        pool = keras.layers.Dense(64, activation="relu", name="dense1")(pool)
        pool = keras.layers.Dropout(0.25)(pool)
        self.CNNOutput = pool

    def setupRNN(self):
        units = [128, 64, 32]
        rnn_input = self.CNNOutput
        for i in range(len(units)):
            rnn_input = keras.layers.Bidirectional(
                keras.layers.LSTM(units[i], return_sequences=True, dropout=0.25), name="Bidirectional_layer" + str(i)
            )(rnn_input)

        self.RNNOutput = keras.layers.Dense(
            len(self.datapreprocessor.char_to_num.get_vocabulary()) + 2, activation="softmax", name="dense2"
        )(rnn_input)

    def setupCTC(self):
        output = CTCLayer(name="ctc_loss")(self.labels, self.RNNOutput)

        model = keras.models.Model(
            inputs=[self.input_img, self.labels], outputs=output, name="handwritting_rec_model"
        )

        opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, clipnorm=1.0)

        model.compile(optimizer=opt)
        model.summary()
        return model

    def train_model(self, epochs):
        val_images = []
        val_labels = []
        early_stopping_patience = 10
        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=early_stopping_patience,
            restore_best_weights=True
        )

        checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath="..\\Model\\handwritting_recognization_model_v2.hdf5",
                                                        monitor='val_loss',
                                                        verbose=1,
                                                        save_best_only=True,
                                                        mode='min')

        wandb.login(key='54c3de5516f53236f9648bf7bdde028c4392b53b')
        wandb.init()
        callbacks_list = [checkpoint,
                          WandbCallback(monitor="val_loss",
                                        mode="min",
                                        log_weights=True),
                          early_stopping
                          ]

        for batch in self.datapreprocessor.valSet:
            val_images.append(batch["image"])
            val_labels.append(batch["label"])
        model = self.setupCTC()
        '''prediction_model = keras.models.Model(
            model.get_layer(name="image").input, model.get_layer(name="dense2").output
        )
         edit_distance_callback = EditDistanceCallback(prediction_model, val_images, val_labels, datapreprocessor)
        '''

        # on charge le model a partir du checkpoint
        if os.path.exists("..\\Model\\handwritting_recognization_model.hdf5"):
            model = tf.keras.models.load_model("..\\Model\\handwritting_recognization_model_v2.hdf5",
                                               custom_objects={'CTCLayer': CTCLayer})

        history = model.fit(
            self.datapreprocessor.trainSet,
            validation_data=self.datapreprocessor.valSet,
            epochs=epochs,
            callbacks=callbacks_list
        )

        # enregistrer le model
        model.save("..\\Model\\handwritting_recognization_model_v2.h5")

        return history
