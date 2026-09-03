from ultralytics import YOLO

model=YOLO("c:\\Users\\thero\\Downloads\\no_plate_model.pt") 

results = model("c:\\ocrTest\\Screenshot 2026-07-20 101723.png", conf=0.4)

for result in results:
    print(result.boxes)
    result.save("output.jpg")