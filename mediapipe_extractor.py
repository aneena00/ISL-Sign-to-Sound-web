import os
import cv2
import mediapipe as mp
import numpy as np

class MediaPipeFeatureExtractor:
    def __init__(self):
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        BaseOptions = mp.tasks.BaseOptions

        self.model_path = "hand_landmarker.task"
        if not os.path.exists(self.model_path):
            print("\n[System] Downloading hand landmark tracking model file...")
            import urllib.request
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
            urllib.request.urlretrieve(url, self.model_path)
            print("[System] Asset download verified successfully!")

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=2
        )
        self.detector = HandLandmarker.create_from_options(options)
        self.hands_result = None

    def extract_features(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        self.hands_result = self.detector.detect(mp_image)
        
        # Initialize a clean 126-dimensional array structured as (2, 21, 3)
        hand_data = np.zeros((2, 21, 3))
        
        if self.hands_result.hand_landmarks and self.hands_result.handedness:
            for hand_landmarks, hand_info in zip(self.hands_result.hand_landmarks, self.hands_result.handedness):
                # Ensure left and right hands match the specific data array indexes (0 and 1)
                hand_type = hand_info[0].category_name
                hand_idx = 0 if hand_type == "Left" else 1
                
                # Extract the 21 joints for this hand
                for j, lm in enumerate(hand_landmarks):
                    hand_data[hand_idx, j, 0] = lm.x
                    hand_data[hand_idx, j, 1] = lm.y
                    hand_data[hand_idx, j, 2] = lm.z
                
                # --- DATASET WRIST NORMALIZATION ---
                # Subtract wrist position (joint 0) from all landmarks on this hand
                wrist = hand_data[hand_idx, 0, :].copy()
                hand_data[hand_idx, :, :] -= wrist
                    
        # Flatten the normalized array back to a 126-dimensional row vector
        return hand_data.flatten()
