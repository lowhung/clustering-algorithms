import numpy as np
from kmeans import Kmeans
import utils
import sklearn
from sklearn.metrics import pairwise_distances_argmin


class ImageQuantizer:

    def __init__(self, b):
        self.b = b

    def quantize_image(self, img):
        # w, h, d = img.shape
        w, h, d = original_shape = tuple(img.shape)

        resized_image = np.reshape(img, (w * h, d))

        model = Kmeans(k = (2**self.b))
        model.fit(resized_image)
        labels = model.predict(resized_image)
        self.means = getattr(model, "means")

        print("Cluster Assignments")
        print(labels)
        return labels
        # quantized_image = np.zeros((w,h,d))
        #
        # label_idx = 0
        # for i in range(w):
        #     for j in range(h):
        #         quantized_image[i][j] = self.means[labels[label_idx]]
        #         label_idx += 1
        # return quantized_image

    def dequantize_image(self, img):
        w, h, d = img.shape
        resized_image = np.reshape(img, (w * h, d))
        original_image = np.zeros(img.shape)

        model = Kmeans(k = (2**self.b))
        model.fit(resized_image)

        labels = model.predict(resized_image)
        self.means = getattr(model, "means")

        label_idx = 0
        for i in range(w):
            for j in range(h):
                original_image[i][j]  = self.means[labels[label_idx]]
                label_idx += 1
        return original_image
