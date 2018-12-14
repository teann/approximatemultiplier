from __future__ import division
from PIL import Image, ImageFilter 
import matplotlib.pyplot as plt 
import cv2
from math import *
import math
import numpy as np
import decimal
import scipy
import random

def truncate(f, n):
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*n)[:n]])

def approx(iter, i):
	x = i
	a = 1 + x/(2**iter)
	t = truncate(a, 3)
	j = 0
	while (j<iter):
		t = float(t) * float(t)
		j += 1
	t = truncate(t, 3)
	return t

def open_image(path):
  newImage = Image.open(path)
  return newImage

# Save Image
def save_image(image, path):
  image.save(path, 'png')

# Create a new image with the given size
def create_image(i, j):
  image = Image.new("RGB", (i, j), "white")
  return image

# Get the pixel from the given image
def get_pixel(image, i, j):
    # Inside image bounds?
    width, height = image.size
    if i > width or j > height:
      return None

    # Get Pixel
    pixel = image.getpixel((i, j))
    return pixel

b = .5
im = Image.open('lenna1.jpg')
im.show()
opixels = im.load()
width, height = im.size
stretched = create_image(width, height)
a = .037
for x in range(width):
	for y in range(height):
		tup1 = int(b * (math.exp(a * opixels[x,y][0]) - 1))
		tup2 = int(b * (math.exp(a * opixels[x,y][1]) - 1))
		tup3 = int(b * (math.exp(a * opixels[x,y][2]) - 1))
		tup = (tup1, tup2, tup3)
		stretched.putpixel((x, y), tup)
stretched.save("stretched.jpg", "JPEG")
stretched.show()
it = 9
demo = create_image(width, height)
for x in range(width):
	for y in range(height):
	#	print a *opixels[x,y][0]
		toExp1 = (a * opixels[x,y][0])
		toExp2 = (a * opixels[x,y][1])
		toExp3 = (a * opixels[x,y][2])
		tup1 = int(b * (float(approx(it, toExp1)) - 1))
		tup2 = int(b * (float(approx(it, toExp2)) - 1))
		tup3 = int(b * (float(approx(it, toExp2)) - 1))
		tup = (tup1, tup2, tup3)
		demo.putpixel((x, y), tup)
demo.save("demo.jpg", "JPEG")
demo.show()
plt.subplot(1,3,1)
img1 = cv2.imread('lenna1.jpg',0)
plt.hist(img1.ravel(),256,[0,256]);
plt.xlabel('Bin Number')
plt.ylabel('Intensity')
plt.title("Original")
plt.subplot(1,3,2)
img2 = cv2.imread('stretched.jpg',0)
plt.hist(img2.ravel(),256,[0,256]);
plt.xlabel('Bin Number')
plt.ylabel('Intensity') 
plt.title("Stretched with full precision")
plt.subplot(1,3,3)
img3 = cv2.imread('demo.jpg',0)
plt.hist(img3.ravel(),256,[0,256]);
plt.xlabel('Bin Number')
plt.ylabel('Intensity')
plt.title("Approximate stretch, 5 iterations, 3 digit truncation")
plt.show()
