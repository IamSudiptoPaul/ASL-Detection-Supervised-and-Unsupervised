# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical Assignment On Supervised And Unsupervised Learning Project
Coursework 002 for: CMP-7058A Artificial Intelligence

Feature Extraction

@author: C102 ( 100525654 , 100525448 )
@date:   11/1/2026

"""
# Import Libraries
import mediapipe as mp
import os, csv

# MediaPipe Setup
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_hands=1
)

# Define File Path
dataset_path = "dataset"
output_file = "extracted_features.csv"

# Header: Label First, Then 21 Landmarks 
header = ['Label'] + [f'lm{i}' for i in range(21)]

# File Writer
f = open(output_file, mode='w', newline='')
writer = csv.writer(f)
writer.writerow(header)

# Feature Extraction
with HandLandmarker.create_from_options(options) as landmarker:
    # Go Through Each Folder
    for folder_name in sorted(os.listdir(dataset_path)):
        folder_path = os.path.join(dataset_path, folder_name)
        
        if os.path.isdir(folder_path):
            print(f"Processing folder: {folder_name}")
            for fn in os.listdir(folder_path):
                if fn.lower().endswith('.jpg'):
                    # Use MediaPipe To Load Images And Detect Hand Lankmarks
                    img = mp.Image.create_from_file(os.path.join(folder_path, fn))
                    result = landmarker.detect(img)
                    
                    row = [folder_name]
                    if result.hand_landmarks:
                        # For Each Landmark, Create X,Y,Z Coordinates 
                        for lm in result.hand_landmarks[0]:
                            row.append(f"{lm.x},{lm.y},{lm.z}")
                    else:
                        # Fill Noise With 0s
                        row.extend(["0.0,0.0,0.0"] * 21)
                    
                    writer.writerow(row)
f.close()
print("CSV saved with 22 columns (Label + 21 landmarks).")