# -*- coding: utf-8 -*-

import numpy as np

from PIL import Image, ImageFont, ImageDraw

class TextBox:
    def __init__(self, text, font: ImageFont.FreeTypeFont, color, background_color):
        self.text = text
        self.background = np.array(background_color, dtype=np.float32)
        if np.sum(np.abs(self.background - color)) == 0:
            raise Exception('Text color is the same as background color')

        text_size = font.getsize(text) #in pixel (width, height)
        self.array = (self.background
                      + np.zeros((text_size[1], text_size[0], 3))).astype(np.float32)

        img = Image.fromarray(self.array.astype(np.uint8))
        drawer = ImageDraw.Draw(img)
        drawer.text((0, 0), text, color, font=font)

        self.array = np.array(img, dtype = np.float32)

        self.fit_to_text() #here u, d, l, and r will be changed

    @property
    def shape(self):
        return self.array.shape

    def __repr__(self):
        if 'left' in self.__dict__:
            return "({}, {}, {}, {}) : {}".format(self.left,
                                                  self.up,
                                                  self.right,
                                                  self.down,
                                                  self.text)
        else:
            return "({}, {}, {}, {}) : {}".format(-1,
                                                  -1,
                                                  -1,
                                                  -1,
                                                  self.text)

    def fit_to_text(self):
        #upper crop
        while np.sum(np.abs(self.array[0] - self.background)) < 1.:
            self.array = self.array[1:]

        #lower crop
        while np.sum(np.abs(self.array[-1] - self.background)) < 1.:
            self.array = self.array[:-1]

        #left crop
        while np.sum(np.abs(self.array[:, 0] - self.background)) < 1.:
            self.array = self.array[:, 1:]

        #right crop
        while np.sum(np.abs(self.array[:, -1] - self.background)) < 1.:
            self.array = self.array[:, :-1]

    def add_to_image(self, img, x, y, borders = False):
        self.left = x
        self.right = x + self.shape[1] - 1 #included
        self.up = y
        self.down  = y + self.shape[0] - 1 #included

        mask = self.array != self.background
        img_crop = img[self.up: self.down + 1, self.left: self.right + 1]
        blended_box = np.where(mask, self.array.astype(np.uint8), img_crop)

        img[self.up: self.down + 1, self.left: self.right + 1] = blended_box

        return img

    def add_borders_to_image(self, img):
        if np.sum(self.background) < 100: #not a lot of light
            print('yolo')
            img[self.up  , self.left: self.right + 1] += 100
            img[self.down, self.left: self.right + 1] += 100
            img[self.up: self.down + 1, self.left]  += 100
            img[self.up: self.down + 1, self.right] += 100

        else:
            img[self.up  , self.left: self.right + 1] //= 2
            img[self.down, self.left: self.right + 1] //= 2
            img[self.up: self.down + 1, self.left]  //= 2
            img[self.up: self.down + 1, self.right] //= 2
        return img