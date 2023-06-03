import onnxruntime
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from depthai_sdk import OakCamera, TwoStagePacket, TextPosition
from depthai_sdk.classes import DetectionPacket
import numpy as np
import cv2
import time
# Load the ONNX model
model_path = 'resnet18_500.onnx'
sess = onnxruntime.InferenceSession(model_path)

# Define your custom class labels
labels = ['T_Down', 'T_Left', 'T_Right', 'T_Up', 'bucket']

def classify(img_path):
    # Load and preprocess the input image
    image = Image.open(img_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image)
    input_tensor = np.expand_dims(input_tensor, axis=0)  # Add batch dimension

    # Run the model to make predictions
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    output = sess.run([output_name], {input_name: input_tensor.astype(np.float32)})

    # Get the predicted class index
    predicted_index = np.argmax(output[0])

    # Get the predicted class label
    predicted_label = labels[predicted_index]

    # Assign the predicted label to the 'Type' variable
    Type = predicted_label

    return Type


def Crop_and_resize(img, x1, y1, x2, y2):
    cropped_img = img[y1:y2, x1:x2]
    resized_img = cv2.resize(cropped_img, (180, 180))
    return resized_img

def calculate_distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

with OakCamera(replay="case1.jpg") as oak:
    color = oak.create_camera('color')
    det = oak.create_nn('best_openvino_2022.1_7shave.blob', color, nn_type='yolo')
    det.config_yolo(2, 4, [10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0, 156.0, 198.0, 373.0, 326.0],
                    {"side52": [0, 1, 2], "side26": [3, 4, 5], "side13": [6, 7, 8]}, 
                    0.5, 0.5)
    Labels = ["T", "Bucket"]

    def cb(packet: DetectionPacket, visualizer):
        n = 0
        frame = packet.frame
        T_coordinates = []
        T_types = []
        safe_regions = []
        bucket_coordinates = []  # List to store bucket coordinates
        unique_T_types = set()  # Keep track of unique T types
        distance_factor = 1.1
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 1
        for det in packet.detections:
            n += 1
            label = Labels[int(det.label)]
            mylist1 = list(det.top_left)
            mylist2 = list(det.bottom_right)
            image = Crop_and_resize(packet.frame, mylist1[0], mylist1[1], mylist2[0], mylist2[1])
            path = "/Digging_Crop/Image" + str(n) + ".jpg"
            cv2.imwrite(path, image)
            Type = classify(path)
            cv2.rectangle(frame, det.top_left, det.bottom_right, (255, 0, 0), 2)
            #cv2.putText(frame, str(label), (mylist1[0] - 5, mylist1[1] - 10),cv2.FONT_HERSHEY_TRIPLEX,0.5,(0, 255, 0), 1,cv2.LINE_AA)
            #cv2.putText(frame, str(Type), (mylist2[0] - 5, mylist2[1] - 10),cv2.FONT_HERSHEY_TRIPLEX,0.5,(0, 0, 255), 1,cv2.LINE_AA)

            if label == "T":
                center_x = (mylist1[0] + mylist2[0]) // 2
                center_y = (mylist1[1] + mylist2[1]) // 2
                T_coordinates.append((center_x, center_y))
                T_types.append(Type)
                unique_T_types.add(Type)
            if label == "Bucket":
            # Handle Bucket detection
                mylist1 = list(det.top_left)
                mylist2 = list(det.bottom_right)
                bucket_coordinates.append((mylist1, mylist2))
                cv2.rectangle(frame, det.top_left, det.bottom_right, (0, 255, 0), 2)  # Draw green rectangle for bucket
                cv2.putText(frame, "Bucket", (mylist1[0], mylist1[1] - 10), font, font_scale, (0, 255, 0), font_thickness, cv2.LINE_AA)  # Write "Bucket" text above the rectangle

        
        if T_types.count("T_Right") >= 2 and T_types.count("T_Left") >= 1:
            # Sort T_coordinates based on x-coordinate
            T_coordinates_sorted = sorted(T_coordinates, key=lambda x: x[0])
            # Calculate the average distance between consecutive T's
            avg_distance = np.mean([calculate_distance(T_coordinates_sorted[i], T_coordinates_sorted[i+1]) for i in range(len(T_coordinates_sorted)-1)])
            # Set a threshold distance for T's in the same safe region
            threshold_distance = avg_distance * distance_factor  # Adjust this factor as needed
            # Initialize the first safe region
            current_region = [T_coordinates_sorted[0]]
            for i in range(1, len(T_coordinates_sorted)):
                if calculate_distance(T_coordinates_sorted[i], current_region[-1]) <= threshold_distance:
                    # Add T to the current safe region
                    current_region.append(T_coordinates_sorted[i])
                else:
                    # Start a new safe region
                    safe_regions.append(current_region)
                    current_region = [T_coordinates_sorted[i]]
            # Add the last safe region
            safe_regions.append(current_region)
            
        elif T_types.count("T_Up") >= 1 and T_types.count("T_Down")  >= 1 and T_types.count("T_Left") >= 1:
            # Sort T_coordinates based on x-coordinate
            T_coordinates_sorted = sorted(T_coordinates, key=lambda x: x[0])
            # Calculate the average distance between consecutive T's
            avg_distance = np.mean([calculate_distance(T_coordinates_sorted[i], T_coordinates_sorted[i+1]) for i in range(len(T_coordinates_sorted)-1)])
            # Set a threshold distance for T's in the same safe region
            threshold_distance = avg_distance * distance_factor  # Adjust this factor as needed
            # Initialize the first safe region
            current_region = [T_coordinates_sorted[0]]
            for i in range(1, len(T_coordinates_sorted)):
                if calculate_distance(T_coordinates_sorted[i], current_region[-1]) <= threshold_distance:
                    # Add T to the current safe region
                    current_region.append(T_coordinates_sorted[i])
                else:
                    # Start a new safe region
                    safe_regions.append(current_region)
                    current_region = [T_coordinates_sorted[i]]
            # Add the last safe region
            safe_regions.append(current_region)
        elif T_types.count("T_Up")  >= 1 and T_types.count("T_Down") >= 1 and T_types.count("T_Right") >= 1:
            # Sort T_coordinates based on x-coordinate
            T_coordinates_sorted = sorted(T_coordinates, key=lambda x: x[0])
            # Calculate the average distance between consecutive T's
            avg_distance = np.mean([calculate_distance(T_coordinates_sorted[i], T_coordinates_sorted[i+1]) for i in range(len(T_coordinates_sorted)-1)])
            # Set a threshold distance for T's in the same safe region
            threshold_distance = avg_distance * distance_factor  # Adjust this factor as needed
            # Initialize the first safe region
            current_region = [T_coordinates_sorted[0]]
            for i in range(1, len(T_coordinates_sorted)):
                if calculate_distance(T_coordinates_sorted[i], current_region[-1]) <= threshold_distance:
                    # Add T to the current safe region
                    current_region.append(T_coordinates_sorted[i])
                else:
                    # Start a new safe region
                    safe_regions.append(current_region)
                    current_region = [T_coordinates_sorted[i]]
            # Add the last safe region
            safe_regions.append(current_region)
        elif T_types.count("T_Left") >= 1 and T_types.count("T_Down")  >= 1 and T_types.count("T_Right")  >= 1:
            # Sort T_coordinates based on x-coordinate
            T_coordinates_sorted = sorted(T_coordinates, key=lambda x: x[0])
            # Calculate the average distance between consecutive T's
            avg_distance = np.mean([calculate_distance(T_coordinates_sorted[i], T_coordinates_sorted[i+1]) for i in range(len(T_coordinates_sorted)-1)])
            # Set a threshold distance for T's in the same safe region
            threshold_distance = avg_distance * distance_factor  # Adjust this factor as needed
            # Initialize the first safe region
            current_region = [T_coordinates_sorted[0]]
            for i in range(1, len(T_coordinates_sorted)):
                if calculate_distance(T_coordinates_sorted[i], current_region[-1]) <= threshold_distance:
                    # Add T to the current safe region
                    current_region.append(T_coordinates_sorted[i])
                else:
                    # Start a new safe region
                    safe_regions.append(current_region)
                    current_region = [T_coordinates_sorted[i]]
            # Add the last safe region
            safe_regions.append(current_region)
        elif T_types.count("T_Left") >= 1 and T_types.count("T_Up") >= 1 and T_types.count("T_Right")  >= 1:
            # Sort T_coordinates based on x-coordinate
            T_coordinates_sorted = sorted(T_coordinates, key=lambda x: x[0])
            # Calculate the average distance between consecutive T's
            avg_distance = np.mean([calculate_distance(T_coordinates_sorted[i], T_coordinates_sorted[i+1]) for i in range(len(T_coordinates_sorted)-1)])
            # Set a threshold distance for T's in the same safe region
            threshold_distance = avg_distance * distance_factor  # Adjust this factor as needed
            # Initialize the first safe region
            current_region = [T_coordinates_sorted[0]]
            for i in range(1, len(T_coordinates_sorted)):
                if calculate_distance(T_coordinates_sorted[i], current_region[-1]) <= threshold_distance:
                    # Add T to the current safe region
                    current_region.append(T_coordinates_sorted[i])
                else:
                    # Start a new safe region
                    safe_regions.append(current_region)
                    current_region = [T_coordinates_sorted[i]]
            # Add the last safe region
            safe_regions.append(current_region)
        
        elif T_types.count("T_Down") >= 1 and T_types.count("T_Left")  >= 1 :
            pass
        else:
    # Ignore cases that are not considered safe zones
            pass
                # Visualize safe zones
        for region in safe_regions:
            updated_T_coordinates = region
            pts = np.array(updated_T_coordinates, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(frame, [pts], (0, 255, 0))
            centroid_x = np.mean(pts[:, 0, 0])
            centroid_y = np.mean(pts[:, 0, 1])
            text_width, text_height = cv2.getTextSize("Safe Zone", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            text_x = int(centroid_x - text_width // 2)
            text_y = int(centroid_y + text_height // 2)
            cv2.putText(frame, "Safe Zone", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.imshow(packet.name, packet.frame)

    oak.visualize(det, callback=cb, fps=True)

    oak.start(blocking=True) 