import tensorflow as tf
from tensorflow import keras
import numpy as np
from typing import Tuple
from datetime import datetime


class Trainer:
    def __init__(self,
                 img_size,
                 num_classes,
                 ckpt_path,
                 train_learning_rate,
                 tune_learning_rate,
                 train_generator,
                 test_generator,
                 train_epochs,
                 tune_epochs,
                 steps_per_epoch,
                 validation_steps):
        self.img_size = img_size
        self.num_classes = num_classes
        self.ckpt_path = ckpt_path
        self.train_learning_rate = train_learning_rate
        self.tune_learning_rate = tune_learning_rate
        self.train_generator = train_generator
        self.test_generator = test_generator
        self.train_epochs = train_epochs
        self.tune_epochs = tune_epochs
        self.steps_per_epoch = steps_per_epoch
        self.validation_steps = validation_steps

        self.model = None
        self.tflite_model = None
        self.history = None

        self.METRICS = ['categorical_accuracy',
                        keras.metrics.Precision(name="precision"),
                        keras.metrics.Recall(name="recall"),
                        ]
        self.early_callback = tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy',
                                                               patience=2)

        self.model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.ckpt_path,
            save_weights_only=True,
            monitor='val_categorical_accuracy',
            mode='max',
            save_best_only=True)

        self._now = datetime.now()
        self._dt_str = self._now.strftime("%d%m%Y_%H%M%S")

    def __call__(self, *args, **kwargs):
        self._build_model()
        self.model.summary()
        # keras.utils.plot_model(self.model, show_shapes=True)
        # self._train_model(self.train_learning_rate, self.train_epochs)
        self.model = keras.models.load_model('/content/drive/MyDrive/KNG/cat_noodles/save_ckpt1/keras/EFNB3_15032023_123926')
        print("\n Tuning" + "." * 10)
        self.model.trainable = True
        self._train_model(self.tune_learning_rate, self.tune_epochs)
        self._save_ckpt()

    def _build_model(self) -> None:
        """
        Build to classify model base on backbone efficientnetB3
        """
        base_model = keras.applications.efficientnet_v2.EfficientNetV2B2(input_shape=(*self.img_size, 3),
                                                                         include_top=False,
                                                                         weights="imagenet")
        base_model.trainable = False
        x = base_model.output
        x = tf.keras.layers.GlobalMaxPooling2D()(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(self.num_classes)(x)
        outputs = tf.keras.layers.Activation('softmax')(x)
        self.model = tf.keras.Model(base_model.input, outputs)

    def _train_model(self, learning_rate, num_epochs) -> None:
        """
        Compile and Fit model
        :param learning_rate: learning rate
        :param num_epochs: number of epochs
        """
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=keras.losses.CategoricalCrossentropy(),
            metrics=self.METRICS,
        )

        self.history = self.model.fit(self.train_generator,
                                      validation_data=self.test_generator,
                                      epochs=num_epochs,
                                      callbacks=[self.early_callback, self.model_checkpoint_callback],
                                      steps_per_epoch=self.steps_per_epoch,
                                      validation_steps=self.validation_steps,
                                      )

    def evaluate(self) -> Tuple[float, float]:
        """
        Evaluate model
        :return: the tuple of loss and accuracy
        """
        test_loss, test_acc = self.model.evaluate(self.test_generator, verbose=1)
        return test_loss, test_acc

    def _save_ckpt(self) -> None:
        """
        Save checkpoint model
        """
        model_save_path = f"{self.ckpt_path}/keras/EFNB3_{self._dt_str}"
        self.model.save(model_save_path)
        self.model.save(f'{model_save_path}.h5')
        print(f"\n The model is saved to {model_save_path}")

    def convert_to_tflite(self) -> None:
        """
        Convert keras model to tflite
        """
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        self.tflite_model = converter.convert()

        # Save tflite model
        tflite_save_path = f"{self.ckpt_path}/tflite/EFNB3_{self._dt_str}"
        with open(tflite_save_path, 'wb') as f:
            f.write(self.tflite_model)

    def check_tflite_model(self) -> None:
        """
        Check tflite model is True or not
        """
        # Create dummy input
        dummy_input = np.ones([1, self.img_size[0], self.img_size[1], 3]) * 1.0

        # Use keras model to predict result
        model_out = self.model.predict(dummy_input)[0]

        # Use tflite model to predict result
        input_details = self.tflite_model.get_input_details()
        output_details = self.tflite_model.get_output_details()
        self.tflite_model.set_tensor(input_details[0]['index'], dummy_input[tf.newaxis, ...])
        self.tflite_model.invoke()
        tflite_out = self.tflite_model.get_tensor(output_details[0]['index'])[0]

        assert np.array_equal(model_out, tflite_out), "Result of keras model is different from tflite model!"
