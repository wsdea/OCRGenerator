import numpy as np
import random

from scipy.ndimage import gaussian_filter
from TextBox import TextBox

class DefaultTransformation:
    def __init__(self):
        raise NotImplementedError('Create your own transformation class'
                                  'which inherits from this class'
                                  'and overwrites self.__init__ and self.apply')

    def apply(self, array: np.ndarray, background) -> np.ndarray:
        raise NotImplementedError('Create your own transformation class'
                                  'which inherits from this class'
                                  'and overwrites self.__init__ and self.apply')

    def __call__(self, box: TextBox) -> TextBox:
        out = self.apply(box.array, box.background)

        if not isinstance(out, np.ndarray) or len(out.shape) != 3 or out.shape[2] != 3:
            raise Exception('out array shape is not (height, width, 3)')

        box.array = out
        return box


class Lambda(DefaultTransformation):
    def __init__(self, fun=lambda array, background: array):
        self.fun = fun

    def apply(self, array, background):
        return self.fun(array, background)


class Pipeline(DefaultTransformation):
    def __init__(self, transformation_list=[]):
        self.transformation_list = transformation_list

    def apply(self, array, background):
        for t in self.transformation_list:
            array = t.apply(array, background)
        return array


class RandomizedPipeline(DefaultTransformation):
    def __init__(self, transformation_list=[]):
        self.transformation_list = transformation_list

    def apply(self, array, background):
        order = np.random.permutation(len(self.transformation_list))
        for i in order:
            array = self.transformation_list[i].apply(array)
        return array


class CoinFlip(DefaultTransformation):
    def __init__(self, transformation, true_probability=0.5):
        self.p = true_probability
        self.transformation = transformation

    def apply(self, array, background):
        if random.random() < self.p:
            array = self.transformation.apply(array, background)

        return array


class FitToText(DefaultTransformation):
    def __init__(self):
        pass

    def apply(self, array, background):
        #upper crop
        while np.sum(np.abs(array[0] - background)) < 1.:
            array = array[1:]

        #lower crop
        while np.sum(np.abs(array[-1] - background)) < 1.:
            array = array[:-1]

        #left crop
        while np.sum(np.abs(array[:, 0] - background)) < 1.:
            array = array[:, 1:]

        #right crop
        while np.sum(np.abs(array[:, -1] - background)) < 1.:
            array = array[:, :-1]

        return array


class GaussianNoise(DefaultTransformation):
    def __init__(self, sigma):
        self.sigma = sigma / np.sqrt(255.)

    def apply(self, array, background):
        noise = 255 * np.random.normal(0., self.sigma, size=array.shape)
        array = np.where(np.abs(array - background) >= 1, array + noise, array)
        array = array.clip(0, 255)
        return array


class _GaussianBlurUnfit(DefaultTransformation):
    def __init__(self, sigma, truncate=4.0):
        self.sigma = sigma
        self.truncate = truncate

    def apply(self, array, background):
        kernel_half_size = int(self.truncate * self.sigma + 0.5)
        out = []
        for c in range(array.shape[2]):
            input_channel = np.pad(array[:, :, c],
                                   pad_width=kernel_half_size,
                                   mode='constant',
                                   constant_values=background[c])
            out.append(gaussian_filter(input_channel,
                                       sigma=self.sigma,
                                       truncate=self.truncate,
                                       mode='constant',
                                       cval=background[c]))

        return np.stack(out, axis=(-1))


def GaussianBlur(sigma, truncate=4.0):
    return Pipeline([_GaussianBlurUnfit(sigma, truncate),
                     FitToText()])
















