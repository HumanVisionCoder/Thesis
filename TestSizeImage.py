from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from skimage.io import imread
from glob import glob


# open method used to open different extension image file
im = Image.open(r"KN5JAN signer 1\frame0.jpg").resize((50,50)) 
  
# This method will show image in any image viewer 
im.show()