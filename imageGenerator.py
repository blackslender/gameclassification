import pylab
import imageio
import os
import glob
import cv2

IMG_COUNT = 150
IMG_WIDTH = 160
IMG_HEIGHT = 90
dataindex = 0
for filename in glob.glob("./videos/*.mp4"):
  # Directory
  d = filename[:-4]
  print("Processing " + d)
  if not os.path.exists(d):
    os.makedirs(d)
  
  vid = imageio.get_reader(filename,  'ffmpeg')
  num_frames = vid._meta['nframes']
  for i in range(IMG_COUNT):
      print(str(i) + '/' + str(IMG_COUNT))
      frameIdx = num_frames//IMG_COUNT*i
      image = vid.get_data(frameIdx)
      image = cv2.resize(image,(int(IMG_WIDTH),int(IMG_HEIGHT)))
      cv2.imwrite(d+'/' + str(dataindex) + '.jpg',image)
      dataindex += 1