from huggingface_hub import hf_hub_download
from ultralytics import YOLO
import cv2, os
import easyocr
import matplotlib.pyplot as plt
from fast_plate_ocr import LicensePlateRecognizer
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

folder_path = 'C:\\Users\\filip\\Documents\\My Projects\\Python\\Licence_Plates_Identifier\\dataset\\lp_data\\datasets\\all_data'
image_number = 2

image_name = os.listdir(folder_path)[image_number]

img = cv2.imread(os.path.join(folder_path,image_name))

weights = hf_hub_download("yasirfaizahmed/license-plate-object-detection", "best.pt")
yolo_model = YOLO(weights)
result = yolo_model.predict(source=img, device=device)

annotated_frame = result[0].plot(conf=False, labels=False)

license_plate_recognizer = LicensePlateRecognizer("cct-s-v2-global-model")

plates_recognized = []

for r in result[0].boxes:

    if len(r.xyxy) == 0:
        print("No license plate detected.")
        quit()

    x1, y1, x2, y2 = r.xyxy[0]

    cropped_image = img[int(y1):int(y2), int(x1):int(x2)]

    scale = 120 / cropped_image.shape[0]
    cropped_image = cv2.resize(cropped_image, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_CUBIC)

    result = license_plate_recognizer.run(cropped_image)
    plates_recognized.append(result)

    print(result)
    # imgplot_crop = plt.imshow(cropped_image)
    # plt.axis('off')
    # plt.show()

imgplot = plt.imshow(annotated_frame)
plt.axis('off')
plt.show()