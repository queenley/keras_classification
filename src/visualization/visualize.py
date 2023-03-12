import matplotlib.pyplot as plt
import numpy as np


class Visualize:
    def __init__(self,
                 list_pred_array,
                 list_true_label,
                 list_pd_img,
                 class_names):
        self.list_pred_array = list_pred_array
        self.list_true_label = list_true_label
        self.list_pd_img = list_pd_img
        self.class_names = class_names

        self.num_class = len(self.class_names)

    def __call__(self, *args, **kwargs):
        num_img = len(self.list_pd_img)
        num_col = 2
        num_row = num_img // num_col + num_img % num_col
        plt.figure(figsize=(2 * 2 * num_col, 2 * num_row))
        for idx in range(num_img):
            plt.subplot(num_row, 2 * num_col, 2 * idx + 1)
            self._plot_image(self.list_pd_img, self.list_pred_array[idx], self.list_true_label[idx])
            plt.subplot(num_row, 2*num_col, 2*idx+1)
            self._plot_value_array(self.list_pred_array[idx], self.list_true_label[idx])
        plt.tight_layout()
        plt.show()

    def _plot_image(self, img, pred_array, true_label):
        """
        Plot image
        """
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])

        plt.imshow(img, cmap=plt.cm.binary)

        predict_label = np.argmax(pred_array)
        if predict_label == true_label:
            color = 'green'
        else:
            color = 'red'

        plt.xlabel("{} {:2.0f}% ({})".format(self.class_names[predict_label],
                                             100 * np.max(pred_array),
                                             self.class_names[true_label]),
                   color=color)

    def _plot_value_array(self, pred_array, true_label):
        """
        Plot predict result of image
        """
        plt.grid(False)
        plt.xticks(range(self.num_class))
        plt.yticks([])
        thisplot = plt.bar(range(self.num_class), pred_array, true_label)
        plt.ylim([0, 1])
        predict_label = np.argmax(pred_array)

        thisplot[predict_label].set_color('red')
        thisplot[true_label].set_color('green')
