from huggingface_hub import hf_hub_download
from ultralytics import YOLO
import cv2, os
import matplotlib.pyplot as plt
from fast_plate_ocr import LicensePlateRecognizer
import torch


def draw_box(image, x1, y1, x2, y2, thickness):
    """Caixa vermelha com contorno branco por baixo (visível mesmo em carros vermelhos)."""
    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), thickness + 2)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), thickness)  # BGR: vermelho


def draw_label(image, text, x1, y1, y2, font_scale, thickness):
    """Escreve o texto sobre um retângulo sólido, com cores que contrastam com a zona."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    pad = 4
    h_img, w_img = image.shape[:2]
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    label_w = tw + 2 * pad
    label_h = th + baseline + 2 * pad

    top = y1 - label_h - 2
    if top < 0:
        top = y2 + 2
    bottom = top + label_h

    left = max(0, min(x1, w_img - label_w))
    right = left + label_w

    region = image[max(top, 0):min(bottom, h_img), left:right]
    if region.size > 0:
        b, g, r = region.reshape(-1, 3).mean(axis=0)
        luminance = 0.114 * b + 0.587 * g + 0.299 * r
    else:
        luminance = 0

    if luminance > 127:  
        bg_color, text_color = (0, 0, 0), (255, 255, 255)
    else:             
        bg_color, text_color = (255, 255, 255), (0, 0, 0)

    cv2.rectangle(image, (left, top), (right, bottom), bg_color, -1)
    cv2.putText(image, text, (left + pad, bottom - pad - baseline),
                font, font_scale, text_color, thickness, cv2.LINE_AA)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

folder_path = 'C:\\Users\\filip\\Documents\\My Projects\\Python\\Licence_Plates_Identifier\\dataset\\lp_data\\datasets\\all_data'
image_number = 2

image_name = os.listdir(folder_path)[image_number]
img = cv2.imread(os.path.join(folder_path, image_name))

weights = hf_hub_download("yasirfaizahmed/license-plate-object-detection", "best.pt")
yolo_model = YOLO(weights)
yolo_result = yolo_model.predict(source=img, device=device)
boxes = yolo_result[0].boxes

if len(boxes) == 0:
    print("No license plate detected.")
    quit()

license_plate_recognizer = LicensePlateRecognizer("cct-s-v2-global-model")

output = img.copy()
font_scale = max(img.shape[1] / 1000, 0.5)
text_thickness = max(1, round(font_scale * 2))
box_thickness = max(1, round(img.shape[1] / 400))

plates_recognized = []

for box in boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])

    cropped_image = img[y1:y2, x1:x2]
    if cropped_image.size == 0:
        continue

    scale = 120 / cropped_image.shape[0]
    cropped_image = cv2.resize(cropped_image, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_CUBIC)

    ocr_result = license_plate_recognizer.run(cropped_image, return_confidence=True)
    text = ocr_result[0].plate

    draw_box(output, x1, y1, x2, y2, box_thickness)
    draw_label(output, text, x1, y1, y2, font_scale, text_thickness)

    plates_recognized.append(((x1, y1, x2, y2), ocr_result[0]))
    print(ocr_result)

plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB)) 
plt.axis('off')
plt.show()