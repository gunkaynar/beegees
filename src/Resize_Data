import PIL
import os
import os.path
import glob
from PIL import Image

root = "/Users/fatmanur/beegees-main/data/"
for filename in glob.iglob(os.path.normpath(os.path.join(root, "**/*.jpg")), recursive = True):
    image = Image.open(filename)
    image = image.resize((224,224))
    image.save(filename)
