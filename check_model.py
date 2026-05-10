from ultralytics import YOLO

# Point this to the file you *think* is your component model
model = YOLO("model_inspector.pt") 

print("\n🧠 MODEL DNA TEST 🧠")
print("This model was trained to find the following things:")
print(model.names)