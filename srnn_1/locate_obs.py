from  tkinter import *
import cv2 as cv
from matplotlib import pyplot as plt
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfilename

def main():
    root = Tk()
    # im = cv.imread('/home/serene/Documents/copy_srnn_pytorch_2/srnn-pytorch-master/data/ewap_dataset/seq_eth/map.png')
    # plt.imshow(im)
    c = Canvas(root , width= 640, height= 480)
    c.pack()
    File = askopenfilename(parent=root, initialdir="./copy_srnn_pytorch_2/srnn-pytorch-master/data", title='Select an image')
    original = Image.open(File)
    im_tk = ImageTk.PhotoImage(original)
    c.create_image(0,0,image = im_tk, anchor="nw")

    c.bind("<Button 1>" , trans_coord)

    # traj_obs = open('/home/serene/Documents/copy_srnn_pytorch/srnn-pytorch-master/data/obstacles_loc.cpkl', 'rb')
    # line = traj_obs.readline()
    root.mainloop()

def trans_coord(event):
    x = event.x / 640 * 2 - 1
    y = event.y / 480 * 2 - 1
    print(y,x)

if __name__ == "__main__":
    main()