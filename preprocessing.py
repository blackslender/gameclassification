"""
@file laplace_demo.py
@brief Sample code showing how to detect edges using the Laplace operator
"""
import sys
import cv2
import glob
import os
import shutil
import json


def rescale():
    for c in glob.glob("data\\*"):
        i = 0
        for file in glob.glob(c+"\\*"):
            i += 1
            print("Procesing ", file)
            img = cv2.imread(file, cv2.IMREAD_COLOR)
            if img is None:
                print('Error opening image, skipping')
                continue
            img = cv2.resize(img, (160, 90))
            # src = cv2.GaussianBlur(src, (1, 1), 0)
            # src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
            # dstLaplacian = cv2.Laplacian(src_gray, ddepth, kernel_size)
            # dstCanny = cv2.Canny(src_gray, 10, 80)
            # dst = dstCanny + 3*dstLaplacian
            cv2.imwrite(c+'_'+str(i)+'.jpg', img)
            print(c+'_'+str(i)+'.jpg' + " Finished!")

    return 0


def labelAndSplit():
    i = 0
    label = dict()
    for filename in glob.glob("data\\*.jpg"):

        # print(os.path.abspath(filename))

        i += 1
        imgType = 'training'
        if i % 5 == 0:
            imgType = 'testing'
        elif i % 5 == 1:
            imgType = 'validating'

        q = filename
        while q[-1] != '_':
            q = q[:-1]
        q = q[5:-9]
        if q[-1] == ' ':
            q = q[:-1]
        if not os.path.exists(os.path.abspath(imgType) + '\\' + q):
            os.mkdir(os.path.abspath(imgType) + '\\' + q)

        newname = os.path.abspath(imgType + '\\' + q + '\\' + str(i) + '.jpg')
        label[newname] = q

        shutil.copy(filename, newname)
        print(newname)
    labelJson = json.dumps(label)
    f = open("labels.json", 'w')
    f.write(labelJson)
    f.close()
    return 0


def main(argv):
    rescale()
    labelAndSplit()


if __name__ == "__main__":
    main(sys.argv[1:])
