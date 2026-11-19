from huggingface_hub import hf_hub_download
from ultralytics import YOLO
import cv2, os
import easyocr
import matplotlib.pyplot as plt
from fast_plate_ocr import LicensePlateRecognizer

folder_path = 'C:\\Users\\filip\\Documents\\My Projects\\Python\\Licence_Plates_Identifier\\dataset\\lp_data\\datasets\\all_data'
images_all = []
image_number = 0

for filename in os.listdir(folder_path):
    img = cv2.imread(os.path.join(folder_path,filename))
    if img is not None:
        images_all.append(img)

weights = hf_hub_download("yasirfaizahmed/license-plate-object-detection", "best.pt")
model = YOLO(weights)
#source = 'C:\\Users\\filip\\Documents\\My Projects\\Python\\Licence_Plates_Identifier\\dataset\\lp_data\\datasets\\all_data\\0cacb08195a3e2d7_jpg.rf.c476bb424132c7c128aff2715b9016e8.jpg'
result = model.predict(source=images_all[image_number])

annotated_frame = result[0].plot()

# imgplot = plt.imshow(annotated_frame)
# plt.axis('off')
# plt.show()

reader = easyocr.Reader(['en']) 
license_plate_recognizer = LicensePlateRecognizer("cct-s-v2-global-model")

for r in result:

    if len(r.boxes.xyxy) == 0:
        print("No license plate detected.")
        quit()

    x1, y1, x2, y2 = r.boxes.xyxy[0]

    x_size = int(x2) - int(x1)
    y_size = int(y2) - int(y1)

    cropped_image = images_all[image_number][int(y1):int(y2), int(x1):int(x2)]

    scale = 120 / cropped_image.shape[0]
    cropped_image = cv2.resize(cropped_image, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_CUBIC)

    result = license_plate_recognizer.run(cropped_image)
    print(result)

    imgplot_crop = plt.imshow(cropped_image)
    plt.axis('off')
    plt.show()