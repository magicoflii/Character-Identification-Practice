#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
from PIL import Image

input_dir = './code_imgs'
output_dir = './splited_imgs'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for (path, dirnames, filenames) in os.walk(input_dir):
    for filename in filenames:
        if filename.endswith('.jpg'):
            print('Being processed picture %s' % filename)
            img_path = path + '/' + filename
            img = Image.open(img_path)
            w, h = img.size

            #filename = filename.replace(' ','')
            #只保留数字和字母
            #filename = filter(str.isalnum, filename)
            fn = filename.split('.')
            fn[0] = filter(str.isalnum, fn[0])
            if len(fn[0]) == 4:
                sub_dir = fn[0]

                for sub_dir_item in sub_dir:
                    if not os.path.exists(os.path.join(output_dir, sub_dir_item)):
                        os.makedirs(os.path.join(output_dir, sub_dir_item))

                box = (3, 0, 17, h)
                img.crop(box).save(os.path.join(output_dir, sub_dir[0], fn[0] + '_1.bmp'), 'bmp')

                box = (15, 0, 29, h)
                img.crop(box).save(os.path.join(output_dir, sub_dir[1], fn[0] + '_2.bmp'), 'bmp')

                box = (27, 0, 41, h)
                img.crop(box).save(os.path.join(output_dir, sub_dir[2], fn[0] + '_3.bmp'), 'bmp')

                box = (39, 0, 53, h)
                img.crop(box).save(os.path.join(output_dir, sub_dir[3], fn[0] + '_4.bmp'), 'bmp')

