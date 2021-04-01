import numpy as np
import os
import random

from nltk.corpus import words
from PIL import ImageFont

from PDFImage import PDFImage
from Transformations import *

class PDFGenerator:
    def __init__(self, pipeline):
        self.FONT_DIR = r'C:\Windows\Fonts'

        self.A4 = 29.7/21
        self.size = 1000
        self.N_BOXES = 50
        self.MAX_WORDS_PER_BOX = 10
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
                                if not x in excluded_fonts and x[-4:].lower() == '.ttf']

        self.vocab = words.words()

        self.pipeline = pipeline

    def generate_new_image(self):

        im = PDFImage((int(self.size * self.A4), self.size),
                      pipeline=self.pipeline,
                      background=(255,255,255))
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
#                                           color='random')
            if fail_reason == "no_more_space":
                break

        return im




if __name__ == '__main__':
    # pipeline = ChoseAny([GaussianBlur(2), GaussianNoise(1), RandomHoles])
    # pipeline = Pipeline([RandomHoles(0.1)])
    pipeline = Pipeline([])
    G = PDFGenerator(pipeline)
    im = G.generate_new_image()

    print(im.box_list)

    im.save_img(True , 'example_borders.png')
    im.save_img(False, 'example.png')










