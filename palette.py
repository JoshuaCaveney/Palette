from sklearn.cluster import MiniBatchKMeans
import argparse
import numpy as np
import cv2
import json
import tempfile


def print_hex(hist, centroids):
    hex = ['#', '#', '#', '#', '#']
    i = 0
    for color in centroids:
        for c in color.astype("uint8").tolist():
            hex[i] += "%0.2X" % c
        i += 1
    foo = dict(zip(hist, hex))
    with open('hex.json', 'w') as outfile:
        json.dump(foo, outfile, sort_keys=True, indent=2)

def centroid_histogram(clt):
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins = numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()

    return hist

def _main():
    # Parses image as argument
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required = True, help = "Path to the image")
    args = vars(ap.parse_args())

    # Assigns image to the file passed to the argument
    image = cv2.imread(args["image"])
    # The image is converted to RGB so the color will be properly displayed
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #
    image = image.reshape((image.shape[0] * image.shape[1], 3))

    clt = MiniBatchKMeans(5)
    clt.fit(image)

    hist = centroid_histogram(clt)

    print_hex(hist, clt.cluster_centers_)

if __name__ == '__main__':
    _main()
