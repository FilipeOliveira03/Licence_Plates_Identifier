from huggingface_hub import hf_hub_download
from ultralytics import YOLO
import cv2, os
import easyocr
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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

# imgplot = plt.imshow(annotated_frame)
# plt.axis('off')
# plt.show()

for r in result:
    x1, y1, x2, y2 = r.boxes.xyxy[0]

    x_size = int(x2) - int(x1)
    y_size = int(y2) - int(y1)


    
    cropped_image = images_all[image_number][int(y1):int(y2), int(x1):int(x2)]

    scale = 120 / cropped_image.shape[0]   # alvo: ~120 px de altura
    cropped_image = cv2.resize(cropped_image, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    gray = clahe.apply(gray)

    gray = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)

    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=2)
    sharp = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)

    reader = easyocr.Reader(['en']) 

    result = reader.readtext(sharp)

    for bbox, text, prob in result:
        print(f'Text: {text} (Trust: {prob:.2f})')

    imgplot_crop = plt.imshow(sharp)
    plt.axis('off')
    plt.show()