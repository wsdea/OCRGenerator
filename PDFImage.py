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
                 max_overlap=0.1,
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
        # self.free_ys = [True] * self.shape[0]
        self.taken_space = np.zeros(self.shape, dtype=bool)

        assert 0 <= max_overlap <= 1
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
            return 'too_big'

        x = int(random.random() * max_x)
        y = int(random.random() * max_y)

        #checking availability

        if np.mean(self.taken_space[y:y + box.shape[0], x:x + box.shape[1]]) <= self.max_overlap:
            self.array = box.add_to_image(self.array, x, y)
            self.box_list.append(box)
            self.taken_space[box.up: box.down, box.left: box.right] = True
            return '' #no failing reasons

        else:
            return "no_more_space"

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