# -*- coding: utf-8 -*-

import numpy as np

from PIL import Image, ImageFont, ImageDraw

def BGRtoBGRA(array):
    mins = array.min(axis=-1)
    alphas = (255. - mins) / 255.
    new_bgr = (array - mins[:, :, None]) / (alphas[:, :, None] + 1e-12)
    alphas *= 255.
    bgra = np.concatenate((new_bgr, alphas[:, :, None]), axis=-1)
    return bgra.round().clip(0., 255.)

class TextBox:
    def __init__(self, text, font: ImageFont.FreeTypeFont, color):
        self.text = text
        self.background = np.array([255, 255, 255], dtype=np.uint8)

        text_size = font.getsize(text) #in pixel (width, height)
        self.array = self.background + np.zeros((text_size[1], text_size[0], 3), dtype=np.uint8)

        img = Image.fromarray(self.array)
        drawer = ImageDraw.Draw(img)
        drawer.text((0, 0), text, color, font=font)

        self.array = np.array(img, dtype=np.float32)

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

    def add_to_image(self, img, x, y, borders = False):
        self.left = x
        self.right = x + self.shape[1] - 1 #included
        self.up = y
        self.down  = y + self.shape[0] - 1 #included

        text_array = BGRtoBGRA(self.array)
        alpha = text_array[:, :, [3]] / 255.
        text_array = text_array[:, :, :3]
        
        img_crop = img[self.up: self.down + 1, self.left: self.right + 1].astype(np.float32)
        blended_box = img_crop * (1 - alpha) + text_array * alpha
        blended_box = blended_box.clip(0, 255).round().astype(np.uint8)
        
        img[self.up: self.down + 1, self.left: self.right + 1] = blended_box

        return img
        
    def add_borders_to_image(self, img):
        img[self.up  , self.left: self.right + 1] = 0
        img[self.down, self.left: self.right + 1] = 0
        img[self.up: self.down + 1, self.left]  = 0
        img[self.up: self.down + 1, self.right] = 0
        return img
    