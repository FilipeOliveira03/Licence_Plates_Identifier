from huggingface_hub import hf_hub_download
from ultralytics import YOLO
import cv2, os

folder_path = 'C:\\Users\\filip\\Documents\\My Projects\\Python\\Licence_Plates_Identifier\\dataset\\lp_data\\datasets\\all_data'
images_all = []
image_number = 3

for filename in os.listdir(folder_path):
    img = cv2.imread(os.path.join(folder_path,filename))
    if img is not None:
        images_all.append(img)

weights = hf_hub_download("yasirfaizahmed/license-plate-object-detection", "best.pt")
model = YOLO(weights)
#source = 'C:\\Users\\filip\\Documents\\My Projects\\Python\\Licence_Plates_Identifier\\dataset\\lp_data\\datasets\\all_data\\0cacb08195a3e2d7_jpg.rf.c476bb424132c7c128aff2715b9016e8.jpg'
result = model.predict(source=images_all[image_number])

annotated_frame = result[0].plot()

cv2.imshow("Detecção de Matrículas", annotated_frame)
cv2.waitKey(0) 
cv2.destroyAllWindows()