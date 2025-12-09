from ultralytics import YOLO
import os
from PIL import Image

if __name__ == "__main__":
    # Training a YOLOv8 model with combined datasets

    # Path to data.yaml
    yaml_path = "dataset_combined.yaml"


    # Load base YOLOv8 model
    model = YOLO("yolov8n.pt")  

    # Training
    model.train(
        data=yaml_path,
        epochs=50,
        imgsz=640,
        batch=8,
        project="runs_combined"  # Project folder for saving results
    )

    best_model_path = "detection_model/best.pt" # Path to the best model after training
    print("Model saved at:", best_model_path)
    model = YOLO(best_model_path)


    import os
    import glob
    from PIL import Image

    input_folder = "NOOK_W"  # Carpeta con imágenes originales
    predict_folder = "runs/NOOK_W"  # Carpeta donde YOLO guarda predicciones
    crop_output = "crops_NOOK_W"  # Carpeta donde guardaremos los crops

    os.makedirs(crop_output, exist_ok=True)

    print("\nMaking predictions...")

    results = model.predict(
        source=input_folder,
        save=True,      # Save images with detections
        save_txt=True,  # Save YOLO labels in .txt
        conf=0.25,
        project="runs",
        name="NOOK_W",
    )

    print(f"Predictions completed. Results saved in: {predict_folder}")

    labels_dir = os.path.join(predict_folder, "labels")

    print("\nGenerating crops for each detection...")

    for img_name in os.listdir(predict_folder):
        if not (img_name.endswith(".jpg") or img_name.endswith(".png")):
            continue

        base = img_name.rsplit(".", 1)[0]

        # Search for original image in input_folder with any extension
        matches = glob.glob(os.path.join(input_folder, base + ".*"))
        if not matches:
            print(f"Original image not found for {base}")
            continue

        original_img_path = matches[0]
        img = Image.open(original_img_path)
        W, H = img.size

        # Open label file
        label_path = os.path.join(labels_dir, base + ".txt")
        if not os.path.exists(label_path):
            print(f"Label not found for {base}, skipping...")
            continue

        with open(label_path, "r") as f:
            lines = f.readlines()

        crop_id = 0
        for line in lines:
            cls, x, y, w, h = map(float, line.strip().split())

            # Convert YOLO coordinates (normalized) to pixels
            x_center = x * W
            y_center = y * H
            width = w * W
            height = h * H

            # Crop coordinates, ensuring limits within the image
            margin_x = int(width * 0.05)   # 5% optional margin
            margin_y = int(height * 0.05)

            x1 = max(int(x_center - width / 2) - margin_x, 0)
            y1 = max(int(y_center - height / 2) - margin_y, 0)
            x2 = min(int(x_center + width / 2) + margin_x, W)
            y2 = min(int(y_center + height / 2) + margin_y, H)

            # Crop and save
            crop = img.crop((x1, y1, x2, y2))
            crop.save(f"{crop_output}/{base}_crop{crop_id}.png")
            crop_id += 1

    print("\nCROPS READY in folder:", crop_output)

