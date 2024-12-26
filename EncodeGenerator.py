import cv2
import face_recognition
import os
import pickle

# Importing student images
folderPath = 'Images'
pathList = os.listdir(folderPath)
print("Found files:", pathList)

imgList = []
studentIds = []

# Filter out non-image files
valid_extensions = (".jpg", ".jpeg", ".png")  # Add more if needed
filteredList = [file for file in pathList if file.lower().endswith(valid_extensions)]
print("Filtered files:", filteredList)

# Load images and student IDs
for path in filteredList:
    img_path = os.path.join(folderPath, path)
    img = cv2.imread(img_path)

    if img is None:
        print(f"Warning: Could not read image {img_path}")
        continue

    imgList.append(img)
    studentIds.append(os.path.splitext(path)[0])

print("Student IDs:", studentIds)

def findEncodings(imagesList):
    encodeList = []
    for idx, img in enumerate(imagesList):
        try:
            # Convert to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Get face encodings
            encodings = face_recognition.face_encodings(img)
            if len(encodings) > 0:
                encodeList.append(encodings[0])  # Append the first face encoding
            else:
                print(f"Warning: No face detected in image {idx}")
        except Exception as e:
            print(f"Error processing image {idx}: {e}")

    return encodeList

print("Encoding Started ...")
encodeListKnown = findEncodings(imgList)
encodeListKnownWithIds = [encodeListKnown,studentIds]
print(encodeListKnown)
print("Encoding Complete")

file = open("EncodeFile.p", 'wb')         
pickle.dump(encodeListKnownWithIds, file)
file.close()
print("File saved")


