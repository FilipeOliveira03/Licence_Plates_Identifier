from huggingface_hub import hf_hub_download
from ultralytics import YOLO

# pick the weights file from this repo's "Files and versions" tab
weights = hf_hub_download("yasirfaizahmed/license-plate-object-detection", "best.pt")
model = YOLO(weights)
source = 'C:\\Users\\filip\\Documents\\My Projects\\Python\\Licence_Plates_Identifier\\dataset\\lp_data\\datasets\\all_data\\0c756c9366a8cb10_jpg.rf.5448d5081af4eddc034b35111e17a035.jpg'
model.predict(source=source, save=False, show=True)