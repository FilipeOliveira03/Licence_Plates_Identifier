from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="ntnxx2/license_plate_roboflow",
    repo_type="dataset",
    local_dir="C:/Users/filip/Documents/My Projects/Python/Licence_Plates_Identifier/dataset/lp_data",
)