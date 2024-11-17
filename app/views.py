import base64
import io
from PIL import Image
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import numpy as np
import cv2
from ultralytics import YOLO
model = YOLO("app/last.pt")
session=None
good=0
bad=0
percent=0

@csrf_exempt
def process_image(request):
    global session,bad,good
    if request.method == "POST":
        body_data = json.loads(request.body.decode('utf-8'))
        image_data = body_data.get("image")
        sessionid = body_data.get("session")
        if session is None:
            session=sessionid
        elif session != sessionid:
            session=None
            good=0
            bad=0
            percent=0
        print(session)
        if image_data:
            # Remove the "data:image/png;base64," prefix if it's present
            image_data = image_data.split(",")[1]
            # Decode the base64 string
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            # Convert Pillow image to OpenCV format (BGR or RGB)
            open_cv_image = np.array(image)
            open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
            results = model(open_cv_image)
            first_result = results[0]
            
            frame = first_result.orig_img.copy()  # Start with the original image
            
            detections = first_result.boxes
            masks = first_result.masks  # Get masks, if available
            classes = detections.cls.cpu().numpy()
            class_result = ""
            
            for i in range(len(classes)):

                class_id = int(classes[i])
                class_name = model.names[class_id]
                box = detections[i].xyxy.cpu().numpy().astype(int)[0]
                if class_name == "Good Welding":
                    good=good+1
                if class_name == "Bad Welding":
                    bad=bad+1
                # Define color based on class name
                color = (0, 255, 0) if class_name == "Good Welding" else (0, 0, 255)
                
                # Draw segmentation mask if it exists
                if masks is not None:
                    # Assuming masks.data holds the array for each mask
                    mask_array = masks.data[i].cpu().numpy()  # Convert mask to NumPy array
                    mask = (mask_array > 0.5).astype(np.uint8)  # Threshold mask for binary display
                    colored_mask = np.zeros_like(frame)
                    colored_mask[mask == 1] = color
                    frame = cv2.addWeighted(frame, 1, colored_mask, 0.5, 0)  # Overlay mask on image
                
                # Draw bounding box and label
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                cv2.putText(frame, class_name, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                class_result += class_name
            
            open_cv_image = frame.astype(np.uint8)
            _, buffer = cv2.imencode('.png', open_cv_image)
            img_str = base64.b64encode(buffer).decode('utf-8')
            
            print(class_result)
            return JsonResponse({"status": str(class_result), "result": img_str})
        
        else:
            return JsonResponse({"status": "error", "message": "No image received"})
    
    return JsonResponse({"status": "error", "message": "Invalid request method"})

def index(request):
    return render(request,"app/video.html")

def get_result(request):
     global bad,good
     total = good+bad 
     if total == 0:
        total = 1 # Prevent division by zero
     percentage= int((good/total)*100)

     return JsonResponse({"good":good,"bad":bad,"percent":percentage})





