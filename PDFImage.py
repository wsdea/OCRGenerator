# -*- coding: utf-8 -*-
import numpy as np
import random
import cv2

from TextBox import TextBox
from Transformations import Pipeline

class PDFImage:
    def __init__(self,
                 shape,
                 pipeline: Pipeline,
                 background=(255, 255, 255),
                 max_overlap=7,
                 min_color_difference=100,
                 ):

        if background == 'random':
            background = np.random.randint(0, 255, 3)
        self.pipeline = pipeline
        self.background = np.array(background, dtype=np.float32)
        self.min_color_difference = min_color_difference
        self.shape = shape
        self.array = (self.background + np.zeros((shape[0], shape[1], 3))).astype(np.uint8)
        self.box_list = []
        self.free_ys = [True] * self.shape[0]
        self.max_overlap = max_overlap

    def choose_random_text_color(self):
        color = np.random.randint(0, 255, 3)
        while np.sqrt(np.sum((color - self.background) ** 2)) < self.min_color_difference:
            color = np.random.randint(0, 255, 3)
            print('too close {} vs {}'.format(self.background, color))
        return tuple(color)

    def add_text_anywhere(self, text, font, color=(0, 0, 0)):
        if color == 'random':
            color = self.choose_random_text_color()

        box = TextBox(text, font, color, self.background)
        box = self.pipeline(box)

        max_x = self.shape[1] - box.shape[1]
        max_y = self.shape[0] - box.shape[0]

        if max_x < 0 or max_y < 0:
            print('Text is too big')
            return True

        x = random.randint(0, max_x)

        y_candidates = [y for y in np.where(self.free_ys)[0]
                            if sum(self.free_ys[y:y + box.shape[0]]) >= box.shape[0] - self.max_overlap
                            and y < max_y] #starting values

        if len(y_candidates) == 0:
            print('No more y space')
            return False

        y = random.choice(y_candidates)

        self.array = box.add_to_image(self.array, x, y)

        self.box_list.append(box)

        self.free_ys[box.up: box.down] = [False] * box.shape[0]

        return True

    def save_img(self, show_borders=False, file_name="img.png"):
        if not file_name.endswith('.png'):
            file_name += file_name + '.png'

        if show_borders:
            array = self.array.copy()
            for b in self.box_list:
                array = b.add_borders_to_image(array)

            cv2.imwrite(file_name, array)
        else:
            cv2.imwrite(file_name, self.array)