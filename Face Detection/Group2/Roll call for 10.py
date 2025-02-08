import os
from deepface import DeepFace
import pandas as pd
from datetime import datetime
import cv2

# dataset
file_path = 'C:\\Users\\Yakamuz\\Desktop\\hozor\\attendance.xlsx'
data = pd.read_excel(file_path)
# add date to exel file
if 'Last Attendance' not in data.columns:
    data['Last Attendance'] = None

# keep recognized faces
already_recognized = set()

# recognition
def recognize_face(image_path):
    try:
        best_match = None  
        best_distance = float('inf')
        
        
        for index, row in data.iterrows():
            #check if student is already present
            if row['Name'] in already_recognized:
                continue
            
            # path in exel
            registered_image = row['Image Path']
            
            # face macth
            result = DeepFace.verify(image_path, registered_image, model_name='VGG-Face')
            
            # better macth
            if result['verified'] and result['distance'] < best_distance:
                best_match = index
                best_distance = result['distance']
                best_name = row['Name']
        
        # best macth write in exel
        if best_match is not None:
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            data.at[best_match, 'Last Attendance'] = current_time
            already_recognized.add(best_name)  # add person to alredy recognized
            print("\n" + f" {best_name} at {current_time}" + "\n")
            return True
        else:
            print("\n" + "No match found!" + "\n")
            return False
    except Exception as e:
        print("Error in face recognition:", e)
        return False

# webcam
def capture_image():
    cam = cv2.VideoCapture(0) 
    cv2.namedWindow("Webcam")
    print("Press SPACE to capture image, or ESC to exit.")
    
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame.")
            break
        cv2.imshow("Webcam", frame)
        
        key = cv2.waitKey(1)
        if key % 256 == 27:  # ESC
            print("end of roll call")
            cam.release()
            cv2.destroyAllWindows()
            return None
        elif key % 256 == 32:  # SPACE 
            img_name = "captured_image.jpg" # image captured
            cv2.imwrite(img_name, frame)
            print(f"Image saved , wait ")
            cam.release()
            cv2.destroyAllWindows()
            return img_name

# loop for more captures
while True:
    print("Opening webcam... Press ESC to stop or SPACE to capture an image.")
    captured_image_path = capture_image()
    
    if captured_image_path:
        if recognize_face(captured_image_path): ##
            data.to_excel(file_path, index=False) # add changes to exel file 
            print("Attendance file updated.")
        else:
            print("No match found, attendance not updated.")
    else:
        print("No image captured. Exiting.")
        break

# final attendance list
present = data[data['Last Attendance'].notna()][['Name', 'Last Attendance']].values.tolist()  
absent = data[data['Last Attendance'].isna()]['Name'].tolist()  

print("\nSummary of Attendance:")
print("Present:")
for name, time in present:
    print(f"- {name} (Time: {time})")

print("\nAbsent:")
for name in absent:
    print(f"- {name}")
