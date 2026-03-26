import tensorflow as tf

from src.models.tsr.deep_learning.deep_learning_models import DLRegressor


class InceptionTimeRegressor(DLRegressor):
    """
    This is a class implementing a single InceptionTime model for time series regression.
    The code is adapted from https://github.com/hfawaz/InceptionTime designed for time series classification.
    Note that this is not the ensemble version
    """

    def __init__(
            self,
            output_directory,
            input_shape,
            verbose=False,
            epochs=1500,
            batch_size=64,
            nb_filters=32,
            use_residual=True,
            use_bottleneck=True,
            depth=6,
            kernel_size=41,
            loss="mean_squared_error",
            metrics=None
    ):
        """
        Initialise the InceptionNetwork model

        Inputs:
            output_directory: path to store results/models
            input_shape: input shape for the models
            verbose: verbosity for the models
            epochs: number of epochs to train the models
            batch_size: batch size to train the models
            nb_filters: number of filters
            use_residuals: boolean indicating to use residuals
            use_bottleneck: boolean indicating to use bottleneck
            depth: depth of the model
            kernel_size: kernel size
            loss: loss function for the models
            metrics: metrics for the models
        """
        self.name = "InceptionTime"

        self.nb_filters = nb_filters
        self.use_residual = use_residual
        self.use_bottleneck = use_bottleneck
        self.depth = depth
        self.kernel_size = kernel_size - 1
        self.bottleneck_size = 32

        super().__init__(
            output_directory=output_directory,
            input_shape=input_shape,
            verbose=verbose,
            epochs=epochs,
            batch_size=batch_size,
            loss=loss,
            metrics=metrics
        )

    def _inception_module(self, input_tensor, stride=1, activation='linear'):
        if self.use_bottleneck and int(input_tensor.shape[-1]) > 1:
            input_inception = tf.keras.layers.Conv1D(filters=self.bottleneck_size,
                                                     kernel_size=1,
                                                     padding='same',
                                                     activation=activation,
                                                     use_bias=False)(input_tensor)
        else:
            input_inception = input_tensor

        # kernel_size_s = [3, 5, 8, 11, 17]
        kernel_size_s = [self.kernel_size // (2 ** i) for i in range(3)]

        conv_list = []

        for i in range(len(kernel_size_s)):
            conv_list.append(tf.keras.layers.Conv1D(filters=self.nb_filters,
                                                    kernel_size=kernel_size_s[i],
                                                    strides=stride,
                                                    padding='same',
                                                    activation=activation,
                                                    use_bias=False)(input_inception))

        max_pool_1 = tf.keras.layers.MaxPool1D(pool_size=3,
                                               strides=stride,
                                               padding='same')(input_tensor)

        conv_6 = tf.keras.layers.Conv1D(filters=self.nb_filters,
                                        kernel_size=1,
                                        padding='same',
                                        activation=activation,
                                        use_bias=False)(max_pool_1)

        conv_list.append(conv_6)

        x = tf.keras.layers.Concatenate(axis=2)(conv_list)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation='relu')(x)
        return x

    def _shortcut_layer(self, input_tensor, out_tensor):
        shortcut_y = tf.keras.layers.Conv1D(filters=int(out_tensor.shape[-1]),
                                            kernel_size=1,
                                            padding='same',
                                            use_bias=False)(input_tensor)
        shortcut_y = tf.keras.layers.BatchNormalization()(shortcut_y)

        x = tf.keras.layers.Add()([shortcut_y, out_tensor])
        x = tf.keras.layers.Activation('relu')(x)
        return x

    def build_model(self, input_shape):
        """
        Build the InceptionNetwork model

        Inputs:
            input_shape: input shape for the model
        """

        input_layer = tf.keras.layers.Input(input_shape)

        x = input_layer
        input_res = input_layer

        for d in range(self.depth):

            x = self._inception_module(x)

            if self.use_residual and d % 3 == 2:
                x = self._shortcut_layer(input_res, x)
                input_res = x

        gap_layer = tf.keras.layers.GlobalAveragePooling1D()(x)

        output_layer = tf.keras.layers.Dense(1, activation='linear')(gap_layer)

        model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss=self.loss,
                      optimizer=tf.keras.optimizers.Adam(),
                      metrics=self.metrics)

        return model
