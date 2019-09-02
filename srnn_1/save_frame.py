import skvideo.io
import cv2 as cv
import os

def main():
    os.chdir('/home/serene/Documents/copy_srnn_pytorch/')
    vidCap = cv.VideoCapture('./data/ewap_dataset/seq_eth/seq_eth.avi')
    _ , frame = vidCap.read()
    # videogen = skvideo.io.vreader('/home/serene/Documents/copy_srnn_pytorch/data/ewap_dataset/seq_eth/seq_eth.avi')
    # frame = cv.cvtColor(videogen.gi_frame, cv.COLOR_BGR2RGB)

    cv.imwrite('./data/eth_background.png' , frame)


if __name__ == '__main__':
    main()


