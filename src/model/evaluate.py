import argparse
from tensorflow.keras.utils import Sequence
from tensorflow import keras
import numpy as np
import cv2
import albumentations as A


class DataGenerator(Sequence):
    def __init__(self,
                 img_paths,
                 CLASS_NAMES,
                 batch_size=256,
                 img_size=(256, 256),
                 n_channels=3,
                 shuffle=True,
                 augmentations=None,
                 ):
        self.img_paths = img_paths
        self.CLASS_NAMES = np.array(CLASS_NAMES)
        self.batch_size = batch_size
        self.img_size = img_size
        self.n_channels = n_channels
        self.n_classes = len(CLASS_NAMES)
        self.shuffle = shuffle
        self.augmentations = augmentations
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.img_paths) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        batch_img_paths = [self.img_paths[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(batch_img_paths)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.img_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_img_paths):
        X = np.empty((self.batch_size, *self.img_size, self.n_channels))
        y = np.empty((self.batch_size, self.n_classes), dtype=int)

        try:
            for i, img_path in enumerate(batch_img_paths):
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                if self.augmentations is not None:
                    # Augment an image
                    transformed = self.augmentations(image=img)
                    img = transformed["image"]

                img = cv2.resize(img, self.img_size, cv2.INTER_AREA)
                label = img_path.split('/')[-2]
                label = (self.CLASS_NAMES == label) * 1

                X[i] = img * 1.0
                y[i] = label
        except:
            print(img_path)
        # Normalize batch data
        X /= 127.5
        X -= 1.

        return X, y


def make_parser():
    parser = argparse.ArgumentParser("Evaluate")

    parser.add_argument("--test_file", required=True, type=str, help="the .txt file contains list image")
    parser.add_argument("--model", required=True, type=str, help="the model path")

    return parser


transform = A.Compose([

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
    A.Rotate(limit=[-90, 90], interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.5),
    A.Rotate(limit=[-180, 180], interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.2)
])

if __name__ == "__main__":
    CLASS_NAMES = ['001', '002', '003', '004', '005', '006', '007', '008',
                   '009', '010', '011', '012', '013', '014', '015', '016',
                   '017', '018', '019', '020', '021', '022', '023', '024']

    args = make_parser()
    with open(args.test_file) as f:
        val_list = [i.strip() for i in f.readlines()]

    val_generator = DataGenerator(val_list, CLASS_NAMES, batch_size=64, augmentations=transform)
    model = keras.models.load_model(args.model)
    out_eval = model.evaluate(val_generator)
    out_predict = model.predict(val_generator)
