import cv2 
from matplotlib import pyplot as plt
import os

folder_path = 'C:\\Users\\filip\\Documents\\My Projects\\Python\\Licence_Plates_Identifier\\dataset\\lp_data\\datasets\\all_data'

images_all = []

for filename in os.listdir(folder_path):
    img = cv2.imread(os.path.join(folder_path,filename))
    if img is not None:
        images_all.append(img)

#print(f"Total images loaded: {len(images_all)}")

plt.imshow(images_all[1])
plt.show()


