import os
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import json

# โหลดโมเดลที่เทรนไว้
model = load_model('/home/saksorn.bu@FUSION.LAB/rubber_ai/Test_model/rubber_leaf_model_best.h5')

# ตั้งชื่อคลาสให้ตรงกับชื่อโฟลเดอร์
class_names = ['Anthracnose', 'Dry_Leaf', 'Healthy', 'Leaf_Spot','Other']
class_to_index = {name: i for i, name in enumerate(class_names)}

# ฟังก์ชันทำนาย
def predict_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)
    return class_index, confidence

# ตรวจสอบภาพทั้งหมดใน val/
val_root = '/home/saksorn.bu@FUSION.LAB/rubber_ai/Test_model/val'

correct = 0
total = 0

for class_name in os.listdir(val_root):
    class_folder = os.path.join(val_root, class_name)
    if not os.path.isdir(class_folder): continue

    for filename in os.listdir(class_folder):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(class_folder, filename)

            pred_index, conf = predict_image(img_path)
            pred_class = class_names[pred_index]

            total += 1
            is_correct = (pred_class == class_name)
            correct += int(is_correct)

            print(f"📷 {filename} | True: {class_name} | Pred: {pred_class} ({conf:.2%}) {'✅' if is_correct else '❌'}")

accuracy = correct / total
print(f"\n✅ Overall Accuracy on val/: {accuracy:.2%} ({correct}/{total})")

#with open('history.json', 'w') as f:
#    json.dump(history.history, f)

# Path ที่ใช้สำหรับเก็บ history.json
history_file_path = '/home/saksorn.bu@FUSION.LAB/history.json'

# โหลด history จากไฟล์
with open(history_file_path, 'r') as f:
    history = json.load(f)