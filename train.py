from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/MFEL-YOLO/MFEL-YOLO.yaml')
    #model = model.load("yolov8s.pt")
    model.train(data='ultralytics/cfg/datasets/AITOD.yaml',batch=8,epochs=70,imgsz=640)
