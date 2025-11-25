Face Verification System 
This notebook implements a face verification pipeline using the DeepFace library. It compares two images and determines whether they belong to the same person using a chosen deep learning model and distance metric.
The notebook installs or imports:
•	DeepFace — main face recognition framework
•	opencv-python (cv2) — image loading, preprocessing, visualization
•	matplotlib — optional visualization
•	numpy — numerical operations
•	pandas — general utilities
•	requests, tqdm, gdown — internal DeepFace dependencies
These are automatically handled when you install DeepFace:
DeepFace verifies the images through:
1.	Face Detection
Finds faces inside input images.
enforce_detection=False means:
If no face is detected, still run on the raw image.
2.	Face Alignment
Aligns faces to a template for consistent comparison.
3.	Embedding Extraction
The chosen model converts the face into a feature vector (embedding).
4.	Distance Calculation
Computes similarity between the two embeddings using the chosen metric.
5.	Thresholding
Each model+metric pair has a predefined threshold.
o	If distance <= threshold → faces match.
o	Otherwise → mismatch.

