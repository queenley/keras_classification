from sklearn.model_selection import train_test_split
import albumentations as A
import numpy as np
from glob import glob
import tensorflow as tf
from PIL import Image


class DataLoader:
    def __init__(self,
                 data_path,
                 batch_size=90,
                 img_size=(256, 256)):
        self.data_path = data_path
        self.batch_size = batch_size
        self.img_size = img_size

        self.train_dataset = dict()
        self.test_dataset = dict()
        self.label2id = dict()

        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0185, scale_limit=0.1, rotate_limit=45, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.5),
            A.RandomSunFlare(flare_roi=(0, 0, 1, 1), src_radius=50, p=0.1),
            A.OneOf([
                A.GaussNoise(),
                A.Blur(blur_limit=3, p=0.3),
                A.MedianBlur(blur_limit=3, p=0.3),
            ], p=0.3),
            A.CLAHE(clip_limit=1.5, p=0.3),
            A.Rotate(limit=(-90, 90), interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False,
                     p=0.5),
            A.Rotate(limit=(-180, 180), interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False,
                     p=0.2)
        ])

    def _split_dataset(self):
        self.train_dataset['images'] = []
        self.train_dataset['labels'] = []
        self.test_dataset['images'] = []
        self.test_dataset['labels'] = []

        for idx, class_path in enumerate(glob(f"{self.data_path}/*")):
            name_class = class_path.split("/")[-1]
            self.label2id[name_class] = idx

            images_path = glob(f"{class_path}/*")
            train, test, _, _ = train_test_split(images_path, images_path, test_size=0.2, random_state=42)
            self.train_dataset['images'] += train
            self.test_dataset['images'] += test

            self.train_dataset['labels'] += [idx] * len(train)
            self.test_dataset['labels'] += [idx] * len(test)

    def _image_pil_preprocessing(self, img_path):
        img = Image.open(img_path.decode("utf-8")).resize(self.img_size).convert("RGB")
        img = np.array(img, dtype="float32") * 1.0
        img = self.transform(image=img)
        img -= 127.5
        img /= 128.
        return img

    def _image_preprocessing(self, input_data):
        img_path = input_data["images"]
        img_path = tf.squeeze(img_path, axis=0)
        img = tf.numpy_function(self._image_pil_preprocessing, [img_path], tf.float32)
        input_data["images"] = img
        return input_data

    def load_dataset(self):
        # Infinity dataset
        train_dataset = tf.data.Dataset.from_tensor_slices(self.train_dataset)
        test_dataset = tf.data.Dataset.from_tensor_slices(self.test_dataset)

        # Shuffle dataset, "I guess the value for buffer_size is about 1/2 of dataset for better shuffle"
        train_dataset = train_dataset.shuffle(buffer_size=len(self.train_dataset['images']) // 2)
        test_dataset = test_dataset.shuffle(buffer_size=len(self.test_dataset['images']) // 2)

        # Prefetch dataset for faster training
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
        test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)
        train_dataset = train_dataset.map(self._image_preprocessing, num_parallel_calls=tf.data.AUTOTUNE)
        test_dataset = test_dataset.map(self._image_preprocessing, num_parallel_calls=tf.data.AUTOTUNE)

        # Add batch size
        train_dataset = train_dataset.batch(self.batch_size)
        test_dataset = test_dataset.batch(self.batch_size)

        return train_dataset.as_numpy_iterator(), test_dataset.as_numpy_iterator()