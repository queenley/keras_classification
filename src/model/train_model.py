import tensorflow as tf
from tensorflow import keras


class Trainer:
 def __init__(self,
              img_size,
              num_classes,
              ckpt_path,
              train_learning_rate,
              tune_learning_rate,
              train_generator,
              test_generator,
              num_epochs):
     self.img_size = img_size
     self.num_classes = num_classes
     self.ckpt_path = ckpt_path
     self.train_learning_rate = train_learning_rate
     self.tune_learning_rate = tune_learning_rate
     self.train_generator = train_generator
     self.test_generator = test_generator
     self.num_epochs = num_epochs

     self.model = None

     self.METRICS = ['categorical_accuracy',
                     keras.metrics.Precision(name="precision"),
                     keras.metrics.Recall(name="recall"),
                     ]
     self.early_callback = tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=2)

     self.model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
         filepath=self.ckpt_path,
         save_weights_only=True,
         monitor='val_categorical_accuracy',
         mode='max',
         save_best_only=True)

 def __call__(self, *args, **kwargs):
     self.build_model()
     self.model.summary()
     self.train_model(self.train_learning_rate)
     self.model.trainable = True
     self.train_model(self.tune_learning_rate)

 def build_model(self):
     base_model = tf.keras.applications.efficientnet.EfficientNetB3(input_shape=(*self.img_size, 3),
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

 def train_model(self, learning_rate):
     self.model.compile(
         optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
         loss='categorical_crossentropy',
         metrics=self.METRICS
     )

     self.model.fit(self.train_generator,
                    validation_data=self.test_generator,
                    epochs=self.num_epochs,
                    callbacks=[self.early_callback, self.model_checkpoint_callback])
