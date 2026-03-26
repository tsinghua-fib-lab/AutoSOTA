import tensorflow as tf

from src.models.tsr.deep_learning.deep_learning_models import DLRegressor


class ResNetRegressor(DLRegressor):
    """
    This is a class implementing the ResNet model for time series regression.
    The code is adapted from https://github.com/hfawaz/dl-4-tsc designed for time series classification.
    """

    def __init__(
            self,
            output_directory,
            input_shape,
            verbose=False,
            epochs=1500,
            batch_size=64,
            loss="mean_squared_error",
            metrics=None
    ):
        """
        Initialise the DL model

        Inputs:
            output_directory: path to store results/models
            input_shape: input shape for the models
            verbose: verbosity for the models
            epochs: number of epochs to train the models
            batch_size: batch size to train the models
            loss: loss function for the models
            metrics: metrics for the models
        """
        self.name = "ResNet"
        super().__init__(
            output_directory=output_directory,
            input_shape=input_shape,
            verbose=verbose,
            epochs=epochs,
            batch_size=batch_size,
            loss=loss,
            metrics=metrics
        )

    def build_model(self, input_shape):
        """
        Build the ResNet model

        Inputs:
            input_shape: input shape for the model
        """
        n_feature_maps = 64
        input_layer = tf.keras.layers.Input(input_shape)

        # BLOCK 1
        conv_x = tf.keras.layers.Conv1D(filters=n_feature_maps,
                                        kernel_size=8,
                                        padding='same')(input_layer)
        conv_x = tf.keras.layers.BatchNormalization()(conv_x)
        conv_x = tf.keras.layers.Activation('relu')(conv_x)

        conv_y = tf.keras.layers.Conv1D(filters=n_feature_maps,
                                        kernel_size=5,
                                        padding='same')(conv_x)
        conv_y = tf.keras.layers.BatchNormalization()(conv_y)
        conv_y = tf.keras.layers.Activation('relu')(conv_y)

        conv_z = tf.keras.layers.Conv1D(filters=n_feature_maps,
                                        kernel_size=3,
                                        padding='same')(conv_y)
        conv_z = tf.keras.layers.BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = tf.keras.layers.Conv1D(filters=n_feature_maps,
                                            kernel_size=1,
                                            padding='same')(input_layer)
        shortcut_y = tf.keras.layers.BatchNormalization()(shortcut_y)

        output_block_1 = tf.keras.layers.add([shortcut_y, conv_z])
        output_block_1 = tf.keras.layers.Activation('relu')(output_block_1)

        # BLOCK 2
        conv_x = tf.keras.layers.Conv1D(filters=n_feature_maps * 2,
                                        kernel_size=8,
                                        padding='same')(output_block_1)
        conv_x = tf.keras.layers.BatchNormalization()(conv_x)
        conv_x = tf.keras.layers.Activation('relu')(conv_x)

        conv_y = tf.keras.layers.Conv1D(filters=n_feature_maps * 2,
                                        kernel_size=5,
                                        padding='same')(conv_x)
        conv_y = tf.keras.layers.BatchNormalization()(conv_y)
        conv_y = tf.keras.layers.Activation('relu')(conv_y)

        conv_z = tf.keras.layers.Conv1D(filters=n_feature_maps * 2,
                                        kernel_size=3,
                                        padding='same')(conv_y)
        conv_z = tf.keras.layers.BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = tf.keras.layers.Conv1D(filters=n_feature_maps * 2,
                                            kernel_size=1,
                                            padding='same')(output_block_1)
        shortcut_y = tf.keras.layers.BatchNormalization()(shortcut_y)

        output_block_2 = tf.keras.layers.add([shortcut_y, conv_z])
        output_block_2 = tf.keras.layers.Activation('relu')(output_block_2)

        # BLOCK 3
        conv_x = tf.keras.layers.Conv1D(filters=n_feature_maps * 2,
                                        kernel_size=8,
                                        padding='same')(output_block_2)
        conv_x = tf.keras.layers.BatchNormalization()(conv_x)
        conv_x = tf.keras.layers.Activation('relu')(conv_x)

        conv_y = tf.keras.layers.Conv1D(filters=n_feature_maps * 2,
                                        kernel_size=5,
                                        padding='same')(conv_x)
        conv_y = tf.keras.layers.BatchNormalization()(conv_y)
        conv_y = tf.keras.layers.Activation('relu')(conv_y)

        conv_z = tf.keras.layers.Conv1D(filters=n_feature_maps * 2,
                                        kernel_size=3,
                                        padding='same')(conv_y)
        conv_z = tf.keras.layers.BatchNormalization()(conv_z)

        # no need to expand channels because they are equal
        shortcut_y = tf.keras.layers.BatchNormalization()(output_block_2)

        output_block_3 = tf.keras.layers.add([shortcut_y, conv_z])
        output_block_3 = tf.keras.layers.Activation('relu')(output_block_3)

        # FINAL
        gap_layer = tf.keras.layers.GlobalAveragePooling1D()(output_block_3)

        output_layer = tf.keras.layers.Dense(1, activation='linear')(gap_layer)

        model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss=self.loss,
                      optimizer=tf.keras.optimizers.Adam(),
                      metrics=self.metrics)

        return model
