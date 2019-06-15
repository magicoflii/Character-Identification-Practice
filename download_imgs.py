#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import sys
from PIL import Image
import requests
import Identify_codes

img_url = 'http://jwgl.hebtu.edu.cn/CheckCode.aspx'
out_path = './identified_imgs'

if not os.path.exists(out_path):
    os.makedirs(out_path)
if not os.path.exists(out_path + '/tmp'):
    os.makedirs(out_path + '/tmp')

def get_image(url, name):
    with open(name, 'wb') as f:
        f.write(requests.get(url).content)
    img = Image.open(name)
    img.save(name)

for i in xrange(0, 100):
    filename = out_path + '/tmp/' + str(i) + '.bmp'
    get_image(img_url, filename)

    img = Image.open(filename)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    result = Identify_codes.identify_image(img)
    img.save(out_path + '/' + ''.join(result) + '.bmp')
    print(''.join(result))