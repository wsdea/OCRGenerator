import numpy as np
import os
import random

#from fontTools.ttLib import TTFont
from nltk.corpus import words
from PIL import Image, ImageFont, ImageDraw

#def get_font_character_set(font_name):
#    with TTFont(os.path.join(FONT_DIR, font_name)) as ttf:
#        chars = []
#        for x in ttf["cmap"].tables:
#            chars += [chr(y[0]) for y in x.cmap.items()]
#
#    chars = list(set(chars))
#    return chars

class PDFGenerator:
    def __init__(self):
        self.FONT_DIR = r'C:\Windows\Fonts'

        self.A4 = 29.7/21
        self.size = 1000
        self.N_BOXES = 100
        self.MAX_WORDS_PER_BOX = 10
        self.FONT_SIZE_EXTREMA = [18, 25]

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

    def generate_new_image(self):
        im = PDFImage((int(self.size * self.A4), self.size),
                      background='random')

        for i in range(self.N_BOXES):
            font_name = random.choice(self.all_fonts)
            font_size = random.randint(*self.FONT_SIZE_EXTREMA)
            font = ImageFont.truetype(font_name, font_size)

            n_words = random.randint(1, self.MAX_WORDS_PER_BOX)
            text = " ".join(np.random.choice(self.vocab, n_words))

            success = im.add_text_anywhere(text, font, color='random')
            if not success:
                break

        print(im.box_list)
        array = np.array(im.img)
        for b in im.box_list:
            array = b.show_borders(array)

        im.img = Image.fromarray(array)
        return im


class PDFImage:
    def __init__(self,
                 shape,
                 background=(255, 255, 255),
                 max_overlap=7,
                 min_color_difference=50,
                 ):

        if background == 'random':
            background = np.random.randint(0, 255, 3)
        self.background = np.array(background, dtype=np.float32)
        self.min_color_difference = min_color_difference
        self.shape = shape
        self.img = (self.background + np.zeros((shape[0], shape[1], 3))).astype(np.uint8)
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

        max_x = self.shape[1] - box.shape[1]
        max_y = self.shape[0] - box.shape[0]

        if max_x < 0 or max_y < 0:
            print('Text is too big')
            return True

        x = random.randint(0, max_x)

        y_candidates = [y for y in np.where(self.free_ys)[0]
                            if sum(self.free_ys[y:y + box.shape[0]]) >= box.shape[0] - self.max_overlap
                            and y < max_y]#starting values

        if len(y_candidates) == 0:
            print('No more y space')
            return False

        y = random.choice(y_candidates)

        self.img = box.add_to_image(self.img, x, y)

        self.box_list.append(box)

        self.free_ys[box.up: box.down] = [False] * box.shape[0]

        return True

    def save(self, *args, **kwargs):
        self.img.save(*args, **kwargs)


class TextBox:
    def __init__(self, text, font, color, background_color):
        self.text = text
        self.background = np.array(background_color, dtype=np.float32)

        text_size = font.getsize(text) #in pixel (width, height)
        self.array = (self.background + np.zeros((text_size[1], text_size[0], 3))).astype(np.uint8)

        img = Image.fromarray(self.array)
        drawer = ImageDraw.Draw(img)
        drawer.text((0, 0), text, color, font=font)

        self.array = np.array(img)
        self.shape = self.array.shape

        self.crop_box_to_image() #here u, d, l, and r will be changed


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

    def crop_box_to_image(self):
        #upper crop
        while np.sum(np.abs(self.array[0] - self.background)) == 0:
            self.array = self.array[1:]

        #lower crop
        while np.sum(np.abs(self.array[-1] - self.background)) == 0:
            self.array = self.array[:-1]

        #left crop
        while np.sum(np.abs(self.array[:, 0] - self.background)) == 0:
            self.array = self.array[:, 1:]

        #right crop
        while np.sum(np.abs(self.array[:, -1] - self.background)) == 0:
            self.array = self.array[:, :-1]

        self.shape = self.array.shape


    def add_to_image(self, img, x, y):
        self.left = x
        self.right = x + self.shape[1] #excluded
        self.up = y
        self.down  = y + self.shape[0] #excluded

        mask = self.array != self.background
        img_crop = img[self.up: self.down, self.left: self.right]
        blended_box = np.where(mask, self.array, img_crop)

        img[self.up: self.down, self.left: self.right] = blended_box

        return img

    def show_borders(self, img):
        img[self.up      , self.left: self.right] //= 2
        img[self.down - 1, self.left: self.right] //= 2
        img[self.up: self.down, self.left]      //= 2
        img[self.up: self.down, self.right - 1] //= 2
        return img

G = PDFGenerator()
im = G.generate_new_image()

im.save('img.png')










