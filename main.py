import numpy as np
import os
import random
import cv2

from nltk.corpus import words
from PIL import ImageFont

from PDFImage import PDFImage
from Transformations import *

def random_crop(array, crop_shape):
    h_a, w_a, _ = array.shape
    h_crop, w_crop = crop_shape
    assert h_crop <= h_a and w_crop <= w_a
    x = int(random.random() * (w_a - w_crop))
    y = int(random.random() * (h_a - h_crop))
    return array[y: y + h_crop, x: x + w_crop]
    
class PDFGenerator:
    def __init__(self, pipeline):
        #pdf constants
        self.A4 = 29.7/21
        self.size = 1000
        self.shape = (int(self.size * self.A4), self.size)
        self.N_BOXES = 50
        self.MAX_WORDS_PER_BOX = 10

        #fonts
        self.FONT_DIR = r'C:\Windows\Fonts'
        self.FONT_SIZE_RANGE = [18, 25]

        excluded_fonts = [
                          'marlett.ttf',
                          'symbol.ttf',
                          'wingding.ttf',
                          'WINGDNG3.TTF',
                          'WINGDNG2.TTF',
                          'REFSPCL.TTF',
                          'BSSYM7.TTF',
                          'webdings.ttf',
                          "OUTLOOK.TTF",
                          'segmdl2.ttf',
                          'holomdl2.ttf',
                          'MTEXTRA.TTF',
                         ]

        self.all_fonts = [x for x in os.listdir(self.FONT_DIR)
                                if x[-4:].lower() == '.ttf' and not x in excluded_fonts]
        
        #text
        self.vocab = words.words()

        #background
        self.TEXTURE_DIR = "textures"
        self.all_textures = [os.path.join(self.TEXTURE_DIR, x) for x in os.listdir(self.TEXTURE_DIR)]
        
        
        #transformations
        self.pipeline = pipeline

    def generate_new_image(self):
        #background setup
        texture = cv2.imread(random.choice(self.all_textures))
        background = random_crop(texture, self.shape)
        
        im = PDFImage(self.shape,
                      pipeline=self.pipeline,
                      background=background)
                      # background=(255,255,255))
#                      background='random')

        for _ in range(self.N_BOXES):
            font_name = random.choice(self.all_fonts)
            font_size = random.randint(*self.FONT_SIZE_RANGE)
            font = ImageFont.truetype(font_name, font_size)

            n_words = random.randint(1, self.MAX_WORDS_PER_BOX)
            text = " ".join(np.random.choice(self.vocab, n_words))

            fail_reason = im.add_text_anywhere(text,
                                            font,
                                            color=(0,0,0))
                                          # color='random')
            if fail_reason == "no_more_space":
                break

        return im




if __name__ == '__main__':
    pipeline = ChoseAny([GaussianBlur(1.5), GaussianNoise(1), RandomHoles(0.1)])
    # pipeline = ChoseAny([GaussianBlur(1, 1), GaussianBlur(1, 2),GaussianBlur(1 , 3),GaussianBlur(1, 4)])
    # pipeline = Pipeline([GaussianBlur(2)])
    # pipeline = Pipeline([])
    G = PDFGenerator(pipeline)
    im = G.generate_new_image()

    print(im.box_list)

    im.save_img(True , 'example_borders.png')
    im.save_img(False, 'example.png')
