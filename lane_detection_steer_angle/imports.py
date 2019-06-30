import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
from scipy import ndimage
from scipy import signal
from scipy import misc
import math
import cv2
from scipy.signal import argrelextrema
from scipy.ndimage.filters import gaussian_filter1d
from skimage.feature import peak_local_max
from operator import itemgetter
import copy
from numpy.linalg import inv
from math import atan, degrees, sin
from skimage import color
import pickle
from skimage import img_as_float
from PIL import Image, ImageDraw, ImageFont
from collections import deque