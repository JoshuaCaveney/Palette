from sklearn.cluster import KMeans, MiniBatchKMeans
import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
#import utils
import cv2

def main():
    # Parses image as argument
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required = True, help = "Path to the image")
    args = vars(ap.parse_args())

    # Assigns image to the file passed to the argument
    image = cv2.imread(args["image"])
    # The image is converted to RGB so the color will be properly displayed
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Adds the read image to the plot
    '''plt.figure()
    plt.axis("off")
    plt.imshow(image)
    '''
    image = image.reshape((image.shape[0] * image.shape[1], 3))

    '''timeclt = time.time()
    clt = KMeans(5)
    print("KMeans took ", time.time() - timeclt, " seconds to execute")'''

    timeclt2 = time.time()
    clt2 = MiniBatchKMeans(5)
    print("MiniBatchKMeans took ", time.time() - timeclt2, " seconds to execute")

    '''fit = time.time()
    clt.fit(image)
    print("fitting took ", time.time() - fit, " secs")'''

    fit2 = time.time()
    clt2.fit(image)
    print("fitting took ", time.time() - fit2, " secs")


    #hist = centroid_histogram(clt)
    hist2 = centroid_histogram(clt2)

    #bar = plot_colors(hist, clt.cluster_centers_)
    #print_hex(hist, clt.cluster_centers_)

    '''plt.figure()
    plt.axis('off')
    plt.imshow(bar)'''

    bar2 = plot_colors(hist2, clt2.cluster_centers_)
    print_hex(hist2, clt2.cluster_centers_)

    plt.figure()
    plt.axis('off')
    plt.imshow(bar2)

    plt.show()

def print_hex(hist, centroids):
    hex = ['#', '#', '#', '#', '#']
    i = 0
    for color in centroids:
        for c in color.astype("uint8").tolist():
            hex[i] += "%0.2X" % c
        i += 1
    foo = zip(hist, hex)
    print(sorted(foo, reverse = True))

def centroid_histogram(clt):
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins = numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()

    return hist

def plot_colors(hist, centroids):
    bar = np.zeros((50, 300, 3), dtype = "uint8")
    startX = 0

    for (percent, color) in zip(hist, centroids):
        endX = startX + percent * 300
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
            color.astype("uint8").tolist(), -1)
        startX = endX

    return bar

if __name__ == '__main__':
    main()
