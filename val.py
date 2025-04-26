from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('runs/detect/train/weights/best.pt')
    metrics = model.val(imgsz=640,save_json=True,save_txt=True)  # ,save_dir=''
