import cv2
import numpy as np
file = open("filenames.txt", "r")

blue = []
green = []
red = []

for line in file:
    img_name = line[:-1]
    img_path = f"bkk-urbanscapes-complete/train/{img_name}"
    img = cv2.imread(img_path)
    b = img[:,:,0].flatten().tolist()
    g = img[:,:,1].flatten().tolist()
    r = img[:,:,2].flatten().tolist()
    blue += b
    green += g
    red += r

blue = np.array(blue)
green = np.array(green)
red = np.array(red)

print(f"std blue: {np.std(blue)}")
print(f"std green: {np.std(green)}")
print(f"std red: {np.std(red)}")

print(f"mean blue: {np.mean(blue)}")
print(f"mean green: {np.mean(green)}")
print(f"mean red: {np.mean(red)}")


file.close()