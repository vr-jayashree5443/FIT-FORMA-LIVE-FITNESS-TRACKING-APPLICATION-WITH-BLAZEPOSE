from flask import Flask, render_template, request, redirect, session
import pymysql
from datetime import datetime
from flask import Flask, Response, render_template
import cv2
import imutils
import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
import datetime
import time
import pickle
import math

from flask import Flask, render_template
import sqlite3
import pandas as pd
from datetime import datetime, timedelta


import warnings
warnings.filterwarnings('ignore')
pm= 0
ps=0
s=0
lc=0
rc=0
before="notcorrect"
now="notcorrect"
# Drawing helpers
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
#-----
# Determine important landmarks for plank
IMPORTANT_LMS = [
    "NOSE",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "RIGHT_ELBOW",
    "LEFT_ELBOW",
    "RIGHT_WRIST",
    "LEFT_WRIST",
    "LEFT_HIP",
    "RIGHT_HIP",
]

# Generate all columns of the data frame

HEADERS = ["label"] # Label column

for lm in IMPORTANT_LMS:
    HEADERS += [f"{lm.lower()}_x", f"{lm.lower()}_y", f"{lm.lower()}_z", f"{lm.lower()}_v"]

#-----
def rescale_frame(frame, percent=50):
    '''
    Rescale a frame from OpenCV to a certain percentage compare to its original frame
    '''
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)


def save_frame_as_image(frame, message: str = None):
    '''
    Save a frame as image to display the error
    '''
    now = datetime.datetime.now()

    if message:
        cv2.putText(frame, message, (50, 150), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
        
    print("Saving ...")
    cv2.imwrite(f"../data/logs/bicep_{now}.jpg", frame)


def calculate_angle(point1: list, point2: list, point3: list) -> float:
    '''
    Calculate the angle between 3 points
    Unit of the angle will be in Degree
    '''
    point1 = np.array(point1)
    point2 = np.array(point2)
    point3 = np.array(point3)

    # Calculate algo
    angleInRad = np.arctan2(point3[1] - point2[1], point3[0] - point2[0]) - np.arctan2(point1[1] - point2[1], point1[0] - point2[0])
    angleInDeg = np.abs(angleInRad * 180.0 / np.pi)

    angleInDeg = angleInDeg if angleInDeg <= 180 else 360 - angleInDeg
    return angleInDeg


def extract_important_keypoints(results, important_landmarks: list) -> list:
    '''
    Extract important keypoints from mediapipe pose detection
    '''
    landmarks = results.pose_landmarks.landmark

    data = []
    for lm in important_landmarks:
        keypoint = landmarks[mp_pose.PoseLandmark[lm].value]
        data.append([keypoint.x, keypoint.y, keypoint.z, keypoint.visibility])
    
    return np.array(data).flatten().tolist()

#-----
class BicepPoseAnalysis:
    def __init__(self, side: str, stage_down_threshold: float, stage_up_threshold: float, peak_contraction_threshold: float, loose_upper_arm_angle_threshold: float, visibility_threshold: float):
        # Initialize thresholds
        self.stage_down_threshold = stage_down_threshold
        self.stage_up_threshold = stage_up_threshold
        self.peak_contraction_threshold = peak_contraction_threshold
        self.loose_upper_arm_angle_threshold = loose_upper_arm_angle_threshold
        self.visibility_threshold = visibility_threshold

        self.side = side
        self.counter = 0
        self.stage = "down"
        self.is_visible = True
        self.detected_errors = {
            "LOOSE_UPPER_ARM": 0,
            "PEAK_CONTRACTION": 0,
        }

        # Params for loose upper arm error detection
        self.loose_upper_arm = False

        # Params for peak contraction error detection
        self.peak_contraction_angle = 1000
        self.peak_contraction_frame = None
    
    def get_joints(self, landmarks) -> bool:
        '''
        Check for joints' visibility then get joints coordinate
        '''
        side = self.side.upper()

        # Check visibility
        joints_visibility = [ landmarks[mp_pose.PoseLandmark[f"{side}_SHOULDER"].value].visibility, landmarks[mp_pose.PoseLandmark[f"{side}_ELBOW"].value].visibility, landmarks[mp_pose.PoseLandmark[f"{side}_WRIST"].value].visibility ]

        is_visible = all([ vis > self.visibility_threshold for vis in joints_visibility ])
        self.is_visible = is_visible

        if not is_visible:
            return self.is_visible
        
        # Get joints' coordinates
        self.shoulder = [ landmarks[mp_pose.PoseLandmark[f"{side}_SHOULDER"].value].x, landmarks[mp_pose.PoseLandmark[f"{side}_SHOULDER"].value].y ]
        self.elbow = [ landmarks[mp_pose.PoseLandmark[f"{side}_ELBOW"].value].x, landmarks[mp_pose.PoseLandmark[f"{side}_ELBOW"].value].y ]
        self.wrist = [ landmarks[mp_pose.PoseLandmark[f"{side}_WRIST"].value].x, landmarks[mp_pose.PoseLandmark[f"{side}_WRIST"].value].y ]

        return self.is_visible
    
    def analyze_pose(self, landmarks, frame):
        '''
        - Bicep Counter
        - Errors Detection
        '''
        self.get_joints(landmarks)

        # Cancel calculation if visibility is poor
        if not self.is_visible:
            return (None, None)

        # * Calculate curl angle for counter
        bicep_curl_angle = int(calculate_angle(self.shoulder, self.elbow, self.wrist))
        if bicep_curl_angle > self.stage_down_threshold:
            self.stage = "down"
        elif bicep_curl_angle < self.stage_up_threshold and self.stage == "down":
            self.stage = "up"
            self.counter += 1
        
        # * Calculate the angle between the upper arm (shoulder & joint) and the Y axis
        shoulder_projection = [ self.shoulder[0], 1 ] # Represent the projection of the shoulder to the X axis
        ground_upper_arm_angle = int(calculate_angle(self.elbow, self.shoulder, shoulder_projection))

        # * Evaluation for LOOSE UPPER ARM error
        if ground_upper_arm_angle > self.loose_upper_arm_angle_threshold:
            # Limit the saved frame
            if not self.loose_upper_arm:
                self.loose_upper_arm = True
                # save_frame_as_image(frame, f"Loose upper arm: {ground_upper_arm_angle}")
                self.detected_errors["LOOSE_UPPER_ARM"] += 1
        else:
            self.loose_upper_arm = False
        
        # * Evaluate PEAK CONTRACTION error
        if self.stage == "up" and bicep_curl_angle < self.peak_contraction_angle:
            # Save peaked contraction every rep
            self.peak_contraction_angle = bicep_curl_angle
            self.peak_contraction_frame = frame
            
        elif self.stage == "down":
            # * Evaluate if the peak is higher than the threshold if True, marked as an error then saved that frame
            if self.peak_contraction_angle != 1000 and self.peak_contraction_angle >= self.peak_contraction_threshold:
                # save_frame_as_image(self.peak_contraction_frame, f"{self.side} - Peak Contraction: {self.peak_contraction_angle}")
                self.detected_errors["PEAK_CONTRACTION"] += 1
            
            # Reset params
            self.peak_contraction_angle = 1000
            self.peak_contraction_frame = None
        
        return (bicep_curl_angle, ground_upper_arm_angle)
#-----
with open('KNN_model.pkl', "rb") as f:
    K = pickle.load(f)
with open("input_scaler.pkl", "rb") as f:
    input_scaler1 = pickle.load(f)
#-----
def generate_frames():
    global lc
    global rc
    VISIBILITY_THRESHOLD = 0.65

    # Params for counter
    STAGE_UP_THRESHOLD = 90
    STAGE_DOWN_THRESHOLD = 120

    # Params to catch FULL RANGE OF MOTION error
    PEAK_CONTRACTION_THRESHOLD = 60

    # LOOSE UPPER ARM error detection
    LOOSE_UPPER_ARM = False
    LOOSE_UPPER_ARM_ANGLE_THRESHOLD = 40

    # STANDING POSTURE error detection
    POSTURE_ERROR_THRESHOLD = 0.7
    posture = "C"

    # Init analysis class
    left_arm_analysis = BicepPoseAnalysis(side="left", stage_down_threshold=STAGE_DOWN_THRESHOLD, stage_up_threshold=STAGE_UP_THRESHOLD, peak_contraction_threshold=PEAK_CONTRACTION_THRESHOLD, loose_upper_arm_angle_threshold=LOOSE_UPPER_ARM_ANGLE_THRESHOLD, visibility_threshold=VISIBILITY_THRESHOLD)

    right_arm_analysis = BicepPoseAnalysis(side="right", stage_down_threshold=STAGE_DOWN_THRESHOLD, stage_up_threshold=STAGE_UP_THRESHOLD, peak_contraction_threshold=PEAK_CONTRACTION_THRESHOLD, loose_upper_arm_angle_threshold=LOOSE_UPPER_ARM_ANGLE_THRESHOLD, visibility_threshold=VISIBILITY_THRESHOLD)
    with mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8) as pose:
        
        while camera.isOpened():
            if camera is not None:
                success, frame = camera.read()
                if not success:
                    continue

                # -----
                # Reduce size of a frame
                #image = rescale_frame(frame, 50)
                image=frame
                # image = cv2.flip(image, 1)
        
                video_dimensions = [image.shape[1], image.shape[0]]

                # Recolor image from BGR to RGB for mediapipe
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                results = pose.process(image)

                if not results.pose_landmarks:
                    print("No human found")
                    continue

                # Recolor image from BGR to RGB for mediapipe
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Draw landmarks and connections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=(244, 117, 66), thickness=2, circle_radius=2), mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=1))

                # Make detection
                try:
                    landmarks = results.pose_landmarks.landmark
                    
                    (left_bicep_curl_angle, left_ground_upper_arm_angle) = left_arm_analysis.analyze_pose(landmarks=landmarks, frame=image)
                    (right_bicep_curl_angle, right_ground_upper_arm_angle) = right_arm_analysis.analyze_pose(landmarks=landmarks, frame=image)

                    # Extract keypoints from frame for the input
                    row = extract_important_keypoints(results, IMPORTANT_LMS)
                    X = pd.DataFrame([row], columns=HEADERS[1:])
                    X = pd.DataFrame(input_scaler1.transform(X))


                    # Make prediction and its probability
                    predicted_class = K.predict(X)[0]
                    prediction_probabilities = K.predict_proba(X)[0]
                    class_prediction_probability = round(prediction_probabilities[np.argmax(prediction_probabilities)], 2)
                    #image = imutils.resize(image, width=520, height=480) 
                    if class_prediction_probability >= POSTURE_ERROR_THRESHOLD:
                        posture = predicted_class

                    # Visualization
                    # Status box
                    cv2.rectangle(image, (0, 0), (500, 40), (0, 128, 0), -1)

                    # Display probability
                    cv2.putText(image, "RIGHT", (15, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(right_arm_analysis.counter) if right_arm_analysis.is_visible else "UNK", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                    # Display Left Counter
                    cv2.putText(image, "LEFT", (95, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(left_arm_analysis.counter) if left_arm_analysis.is_visible else "UNK", (100, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                    rc=rc+right_arm_analysis.counter
                    lc=lc+left_arm_analysis.counter
                    # * Display error
                    # Right arm error
                    cv2.putText(image, "R_PC", (165, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(right_arm_analysis.detected_errors["PEAK_CONTRACTION"]), (160, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(image, "R_LUA", (225, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(right_arm_analysis.detected_errors["LOOSE_UPPER_ARM"]), (220, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                    # Left arm error
                    cv2.putText(image, "L_PC", (300, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(left_arm_analysis.detected_errors["PEAK_CONTRACTION"]), (295, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(image, "L_LUA", (380, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(left_arm_analysis.detected_errors["LOOSE_UPPER_ARM"]), (375, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                    # Lean back error
                    cv2.putText(image, "LB", (460, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(f"{posture}, {predicted_class}, {class_prediction_probability}"), (440, 30), cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)


                    # * Visualize angles
                    # Visualize LEFT arm calculated angles
                    if left_arm_analysis.is_visible:
                        cv2.putText(image, str(left_bicep_curl_angle), tuple(np.multiply(left_arm_analysis.elbow, video_dimensions).astype(int)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                        cv2.putText(image, str(left_ground_upper_arm_angle), tuple(np.multiply(left_arm_analysis.shoulder, video_dimensions).astype(int)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


                    # Visualize RIGHT arm calculated angles
                    if right_arm_analysis.is_visible:
                        cv2.putText(image, str(right_bicep_curl_angle), tuple(np.multiply(right_arm_analysis.elbow, video_dimensions).astype(int)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
                        cv2.putText(image, str(right_ground_upper_arm_angle), tuple(np.multiply(right_arm_analysis.shoulder, video_dimensions).astype(int)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
                    lc = left_arm_analysis.counter
                    rc = right_arm_analysis.counter
                except Exception as e:
                    print(f"Error: {e}")
        
                #cv2.imshow("CV2", image)
                # Encode the frame as JPEG
                ret, buffer = cv2.imencode('.jpg', image)
                frame1 = buffer.tobytes()

                # Yield the frame in byte format
                yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame1 + b'\r\n')
            else:
                break
    
    stop_camera()
#-----
#----------------------------------------------------------------------------------------------
#Squats
IMPORTANT_LMS1 = [
    "NOSE",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_KNEE",
    "RIGHT_KNEE",
    "LEFT_ANKLE",
    "RIGHT_ANKLE"
]
headers1 = ["label"]
for lm in IMPORTANT_LMS1:
    headers1 += [f"{lm.lower()}_x", f"{lm.lower()}_y", f"{lm.lower()}_z", f"{lm.lower()}_v"]

def extract_important_keypoints1(results) -> list:
    '''
    Extract important keypoints from mediapipe pose detection
    '''
    landmarks = results.pose_landmarks.landmark

    data = []
    for lm in IMPORTANT_LMS1:
        keypoint = landmarks[mp_pose.PoseLandmark[lm].value]
        data.append([keypoint.x, keypoint.y, keypoint.z, keypoint.visibility])
    
    return np.array(data).flatten().tolist()

def rescale_frame1(frame, percent=50):
    '''
    Rescale a frame to a certain percentage compare to its original frame
    '''
    width = int(frame.shape[1] )
    height = int(frame.shape[0])
    dim = (width, height)
    print(percent,frame.shape[1], frame.shape[1])
    print(width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

def calculate_distance1(pointX, pointY) -> float:
    '''
    Calculate a distance between 2 points
    '''

    x1, y1 = pointX
    x2, y2 = pointY

    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def analyze_foot_knee_placement1(results, stage: str, foot_shoulder_ratio_thresholds: list, knee_foot_ratio_thresholds: dict, visibility_threshold: int) -> dict:
    '''
    Calculate the ratio between the foot and shoulder for FOOT PLACEMENT analysis
    
    Calculate the ratio between the knee and foot for KNEE PLACEMENT analysis

    Return result explanation:
        -1: Unknown result due to poor visibility
        0: Correct knee placement
        1: Placement too tight
        2: Placement too wide
    '''
    analyzed_results = {
        "foot_placement": -1,
        "knee_placement": -1,
    }

    landmarks = results.pose_landmarks.landmark

    # * Visibility check of important landmarks for foot placement analysis
    left_foot_index_vis = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].visibility
    right_foot_index_vis = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].visibility

    left_knee_vis = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility
    right_knee_vis = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility

    # If visibility of any keypoints is low cancel the analysis
    if (left_foot_index_vis < visibility_threshold or right_foot_index_vis < visibility_threshold or left_knee_vis < visibility_threshold or right_knee_vis < visibility_threshold):
        return analyzed_results
    
    # * Calculate shoulder width
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    shoulder_width = calculate_distance1(left_shoulder, right_shoulder)

    # * Calculate 2-foot width
    left_foot_index = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x, landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]
    right_foot_index = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]
    foot_width = calculate_distance1(left_foot_index, right_foot_index)

    # * Calculate foot and shoulder ratio
    foot_shoulder_ratio = round(foot_width / shoulder_width, 1)

    # * Analyze FOOT PLACEMENT
    min_ratio_foot_shoulder, max_ratio_foot_shoulder = foot_shoulder_ratio_thresholds
    if min_ratio_foot_shoulder <= foot_shoulder_ratio <= max_ratio_foot_shoulder:
        analyzed_results["foot_placement"] = 0
    elif foot_shoulder_ratio < min_ratio_foot_shoulder:
        analyzed_results["foot_placement"] = 1
    elif foot_shoulder_ratio > max_ratio_foot_shoulder:
        analyzed_results["foot_placement"] = 2
    
    # * Visibility check of important landmarks for knee placement analysis
    left_knee_vis = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility
    right_knee_vis = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility

    # If visibility of any keypoints is low cancel the analysis
    if (left_knee_vis < visibility_threshold or right_knee_vis < visibility_threshold):
        print("Cannot see foot")
        return analyzed_results

    # * Calculate 2 knee width
    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
    knee_width = calculate_distance1(left_knee, right_knee)

    # * Calculate foot and shoulder ratio
    knee_foot_ratio = round(knee_width / foot_width, 1)

    # * Analyze KNEE placement
    up_min_ratio_knee_foot, up_max_ratio_knee_foot = knee_foot_ratio_thresholds.get("up")
    middle_min_ratio_knee_foot, middle_max_ratio_knee_foot = knee_foot_ratio_thresholds.get("middle")
    down_min_ratio_knee_foot, down_max_ratio_knee_foot = knee_foot_ratio_thresholds.get("down")

    if stage == "up":
        if up_min_ratio_knee_foot <= knee_foot_ratio <= up_max_ratio_knee_foot:
            analyzed_results["knee_placement"] = 0
        elif knee_foot_ratio < up_min_ratio_knee_foot:
            analyzed_results["knee_placement"] = 1
        elif knee_foot_ratio > up_max_ratio_knee_foot:
            analyzed_results["knee_placement"] = 2
    elif stage == "middle":
        if middle_min_ratio_knee_foot <= knee_foot_ratio <= middle_max_ratio_knee_foot:
            analyzed_results["knee_placement"] = 0
        elif knee_foot_ratio < middle_min_ratio_knee_foot:
            analyzed_results["knee_placement"] = 1
        elif knee_foot_ratio > middle_max_ratio_knee_foot:
            analyzed_results["knee_placement"] = 2
    elif stage == "down":
        if down_min_ratio_knee_foot <= knee_foot_ratio <= down_max_ratio_knee_foot:
            analyzed_results["knee_placement"] = 0
        elif knee_foot_ratio < down_min_ratio_knee_foot:
            analyzed_results["knee_placement"] = 1
        elif knee_foot_ratio > down_max_ratio_knee_foot:
            analyzed_results["knee_placement"] = 2
    
    return analyzed_results
with open("L1.pkl", "rb") as f:
    L = pickle.load(f)

def generate_frames1():
    global s
    counter = 0
    current_stage = ""
    PREDICTION_PROB_THRESHOLD = 0.7
    VISIBILITY_THRESHOLD = 0.6
    FOOT_SHOULDER_RATIO_THRESHOLDS = [1.2, 2.8]
    KNEE_FOOT_RATIO_THRESHOLDS = {
        "up": [0.5, 1.0],
        "middle": [0.7, 1.0],
        "down": [0.7, 1.1],
    }

    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while camera.isOpened():
            if camera is not None:
                ret, image = camera.read()

                if not ret:
                    continue
                
                # Reduce size of a frame
                image = rescale_frame1(image, 50)

                # Recolor image from BGR to RGB for mediapipe
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                results = pose.process(image)
                if not results.pose_landmarks:
                    continue

                # Recolor image from BGR to RGB for mediapipe
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Draw landmarks and connections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=(244, 117, 66), thickness=2, circle_radius=2), mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=1))
            # Make detection
                try:
                    # * Model prediction for SQUAT counter
                    # Extract keypoints from frame for the input
                    row = extract_important_keypoints1(results)
                    X = pd.DataFrame([row], columns=headers1[1:])

                    # Make prediction and its probability
                    predicted_class = L.predict(X)[0]
                    predicted_class = "down" if predicted_class == 0 else "up"
                    prediction_probabilities = L.predict_proba(X)[0]
                    prediction_probability = round(prediction_probabilities[prediction_probabilities.argmax()], 2)

                    # Evaluate model prediction
                    if predicted_class == "down" and prediction_probability >= PREDICTION_PROB_THRESHOLD:
                        current_stage = "down"
                    elif current_stage == "down" and predicted_class == "up" and prediction_probability >= PREDICTION_PROB_THRESHOLD: 
                        current_stage = "up"
                        counter += 1

                    # Analyze squat pose
                    analyzed_results = analyze_foot_knee_placement1(results=results, stage=current_stage, foot_shoulder_ratio_thresholds=FOOT_SHOULDER_RATIO_THRESHOLDS, knee_foot_ratio_thresholds=KNEE_FOOT_RATIO_THRESHOLDS, visibility_threshold=VISIBILITY_THRESHOLD)

                    foot_placement_evaluation = analyzed_results["foot_placement"]
                    knee_placement_evaluation = analyzed_results["knee_placement"]
                    
                    # * Evaluate FOOT PLACEMENT error
                    if foot_placement_evaluation == -1:
                        foot_placement = "UNK"
                    elif foot_placement_evaluation == 0:
                        foot_placement = "Correct"
                    elif foot_placement_evaluation == 1:
                        foot_placement = "Too tight"
                    elif foot_placement_evaluation == 2:
                        foot_placement = "Too wide"
                    
                    # * Evaluate KNEE PLACEMENT error
                    if knee_placement_evaluation == -1:
                        knee_placement = "UNK"
                    elif knee_placement_evaluation == 0:
                        knee_placement = "Correct"
                    elif knee_placement_evaluation == 1:
                        knee_placement = "Too tight"
                    elif knee_placement_evaluation == 2:
                        knee_placement = "Too wide"
                
                    # Visualization
                    # Status box
                    cv2.rectangle(image, (0, 0), (500, 40), (0,128,0), -1)

                    # Display class
                    cv2.putText(image, "COUNT", (10, 12), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, f'{str(counter)}, {predicted_class}, {str(prediction_probability)}', (7, 30), cv2.FONT_HERSHEY_COMPLEX, .6, (255, 255, 255), 2, cv2.LINE_AA)

                    # Display Foot and Shoulder width ratio
                    cv2.putText(image, "FOOT", (180, 12), cv2.FONT_HERSHEY_COMPLEX, .4, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, foot_placement, (180, 30), cv2.FONT_HERSHEY_COMPLEX, .5, (255, 255, 255), 2, cv2.LINE_AA)

                    # Display knee and Shoulder width ratio
                    cv2.putText(image, "KNEE", (230, 12), cv2.FONT_HERSHEY_COMPLEX, .4, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, knee_placement, (230, 30), cv2.FONT_HERSHEY_COMPLEX, .5, (255, 255, 255), 2, cv2.LINE_AA)

                except Exception as e:
                    print(f"Error: {e}")
            
                ret, buffer = cv2.imencode('.jpg', image)
                frame1 = buffer.tobytes()

                    # Yield the frame in byte format
                yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame1 + b'\r\n')
            else:
                break
    s=counter
    stop_camera()
#--------------------------------------------------------------------------------------------------------------
#plank
IMPORTANT_LMS2 = [
    "NOSE",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_ELBOW",
    "RIGHT_ELBOW",
    "LEFT_WRIST",
    "RIGHT_WRIST",
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_KNEE",
    "RIGHT_KNEE",
    "LEFT_ANKLE",
    "RIGHT_ANKLE",
    "LEFT_HEEL",
    "RIGHT_HEEL",
    "LEFT_FOOT_INDEX",
    "RIGHT_FOOT_INDEX",
]
HEADERS2 = ["label"] # Label column

for lm in IMPORTANT_LMS2:
    HEADERS2 += [f"{lm.lower()}_x", f"{lm.lower()}_y", f"{lm.lower()}_z", f"{lm.lower()}_v"]

def extract_important_keypoints2(results) -> list:
    '''
    Extract important keypoints from mediapipe pose detection
    '''
    landmarks = results.pose_landmarks.landmark

    data = []
    for lm in IMPORTANT_LMS2:
        keypoint = landmarks[mp_pose.PoseLandmark[lm].value]
        data.append([keypoint.x, keypoint.y, keypoint.z, keypoint.visibility])
    
    return np.array(data).flatten().tolist()


def rescale_frame2(frame, percent=50):
    '''
    Rescale a frame to a certain percentage compare to its original frame
    '''
    width = int(frame.shape[1] )
    height = int(frame.shape[0] )
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)
with open("LR_model (4).pkl", "rb") as f:
    sklearn_model = pickle.load(f)

with open("input_scaler (4).pkl", "rb") as f:
    input_scaler = pickle.load(f)
def get_class2(prediction: float) -> str:
    return {
        0: "C",
        1: "H",
        2: "L",
    }.get(prediction)
i=1
time_s=0
def generate_frames2():
    global before
    global now
    global pm
    global ps
    current_stage = ""
    prediction_probability_threshold = 0.6
    
    global time_s
    elapsed_time=0
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while camera.isOpened():
            if camera is not None:
                ret, image = camera.read()
                if not ret:
                    continue

                    # Reduce size of a frame
                image = rescale_frame2(image, 50)
                    # image = cv2.flip(image, 1)

                    # Recolor image from BGR to RGB for mediapipe
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)
                if not results.pose_landmarks:
                    print("No human found")
                    continue

                    # Recolor image from BGR to RGB for mediapipe
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    # Draw landmarks and connections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=(244, 117, 66), thickness=2, circle_radius=2), mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=1))

                    # Make detection
                try:
                        # Extract keypoints from frame for the input
                    row = extract_important_keypoints2(results)
                    X = pd.DataFrame([row], columns=HEADERS2[1:])
                    X = pd.DataFrame(input_scaler.transform(X))

                        # Make prediction and its probability
                    predicted_class = sklearn_model.predict(X)[0]
                    predicted_class = get_class2(predicted_class)
                    prediction_probability = sklearn_model.predict_proba(X)[0]
                        # print(predicted_class, prediction_probability)

                        # Evaluate model prediction
                    if predicted_class == "C" and prediction_probability[prediction_probability.argmax()] >= prediction_probability_threshold:
                        current_stage = "Correct"
                    elif predicted_class == "L" and prediction_probability[prediction_probability.argmax()] >= prediction_probability_threshold: 
                        current_stage = "Low back"
                    elif predicted_class == "H" and prediction_probability[prediction_probability.argmax()] >= prediction_probability_threshold: 
                        current_stage = "High back"
                    else:
                        current_stage = "unk"
                        
                        # Visualization
                        # Status box
                    cv2.rectangle(image, (0, 0), (250, 40), (0,128,0), -1)

                        # Display class
                    cv2.putText(image, "CLASS", (95, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, current_stage, (90, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                        # Display probability
                    cv2.putText(image, "PROB", (15, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(prediction_probability[prediction_probability.argmax()]), (10, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    if current_stage=="Correct" and before not in "Correct" and elapsed_time==0:
                        time_s =time.time()
                        
                    
                except Exception as e:
                    print(f"Error: {e}")
                        
                        
                ret, buffer = cv2.imencode('.jpg', image)
                frame1 = buffer.tobytes()
                before= current_stage
                        # Yield the frame in byte format
                yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame1 + b'\r\n')
               
            else:
                break
    elapsed_time = time.time() - time_s 
    pm=int((elapsed_time % 3600) // 60)
    ps = int(elapsed_time % 60)
    time_s =0
    stop_camera()
#--------------------------------------------------------------------------------------------------------------
camera = None

def start_camera():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)

def stop_camera():
    global camera
    if camera is not None:
        camera.release()
        camera = None
        

 ##-----------------------------------------------------------------------------   
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Set a secret key for session management

# MySQL configuration
mysql_host = 'localhost'
mysql_user = 'root'
mysql_password = 'QWE098*123asd'
mysql_db = 'hi'

# Connect to MySQL
db = pymysql.connect(host=mysql_host, user=mysql_user, password=mysql_password, database=mysql_db, cursorclass=pymysql.cursors.DictCursor)

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        name = request.form['name']
        dob = request.form['DOB']
        date_format = '%m/%d/%Y'
        dob = datetime.strptime(dob, date_format)
        mobile = request.form['mobile']
        password = request.form['password']
        
        # Insert user data into MySQL
        cursor = db.cursor()
        sql = "INSERT INTO logindata (email, name, dob, mobile, password) VALUES (%s, %s, %s, %s, %s)"
        cursor.execute(sql, (email, name, dob, mobile, password))
        db.commit()
        
        # Show a popup or flash message that registration was successful
        return redirect('/')

@app.route('/login', methods=['POST'])
def login():
    if request.method == 'POST':
        email = request.form['username']
        password = request.form['password']
        
        # Check if user exists in MySQL
        cursor = db.cursor()
        sql = "SELECT * FROM logindata WHERE email = %s AND password = %s"
        cursor.execute(sql, (email, password))
        user = cursor.fetchone()
        
        if user:
            session['user'] = user['email']  # Store user's name in session
            return redirect('/page2')
        else:
            return "Invalid username or password"

@app.route('/logout')
def logout():
    session.pop('user', None)  # Clear the 'user' key from session
    return redirect('/')

@app.route('/page2')
def welcome():
    return render_template('page2.html')



@app.route('/track')
def discussion():
    cursor = db.cursor()
    
    
    t = datetime.now().date()
    dynamic_id =session['user']
    '''seven_days_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')

        # Define the SQL query
    sql = """
    SELECT DATE_FORMAT(day1, '%%Y-%%m') AS month_year, SUM(plank_m) AS total_minutes
    FROM exercise
    WHERE id = %s AND day1 >= %s
    GROUP BY DATE_FORMAT(day1, '%%Y-%%m')
"""
        # Execute the query
    cursor.execute(sql, (dynamic_id, seven_days_ago))'''
    
    seven_days_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')

# Define the dates for the last 6 months
    six_months_ago = (datetime.now() - timedelta(days=6*30)).strftime('%Y-%m-%d')

# Define the SQL query for the last 7 days
    sql_query_7_days = """
    SELECT DATE_FORMAT(day1, '%%Y-%%m-%%d') AS day, SUM(plank_m) AS total_minutes
    FROM exercise
    WHERE id = %s AND day1 >= %s
    GROUP BY DATE_FORMAT(day1, '%%Y-%%m-%%d')
"""

# Execute the query for the last 7 days
    cursor.execute(sql_query_7_days, (dynamic_id, seven_days_ago))
    results_7_days = cursor.fetchall()

# Fill in missing days with zeros for the last 7 days
    data_7_days = {}
    for i in range(7):
        date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
        data_7_days[date] = 0

    for row in results_7_days:

        data_7_days[row['day']] = row['total_minutes']
# Define the SQL query for the last 6 months
#----------plank 6 months-------------------
    sql_query_6_months = """
        SELECT DATE_FORMAT(day1, '%%Y-%%m') AS month_year, SUM(plank_m) AS total_minutes
        FROM exercise
        WHERE id = %s AND day1 >= %s
        GROUP BY DATE_FORMAT(day1, '%%Y-%%m')
    """

    current_date = datetime.now()
    six_months_ago = current_date - timedelta(days=6*30)

# Generate a list of all the months in the last 6 months
    months_range = pd.date_range(start=six_months_ago, end=current_date, freq='MS').strftime('%Y-%m').tolist()
    
# Execute the query for the last 6 months
    cursor.execute(sql_query_6_months, (dynamic_id, six_months_ago))
    results_6_months = cursor.fetchall()
    
# Create a dictionary to store the results
    data_6_months = {month: 0 for month in months_range}
    #print(data_6_months)
    #print(results_6_months)
# Update data with results from the query
    for row in results_6_months:
        month_year = row['month_year']
        data_6_months[month_year] = int(row['total_minutes'])
    df = pd.DataFrame()
    
    df['month_year']=data_6_months.keys()
    df['total_minutes']=data_6_months.values()
    
# Close cursor and connection


# Convert month_year to desired format
    df['month_year'] = pd.to_datetime(df['month_year']).dt.strftime('%b %Y')
#---------------squats 6 months-----------------------
    sql_query_6_months1 = """
        SELECT DATE_FORMAT(day1, '%%Y-%%m') AS month_year, SUM(squats_count) AS total_minutes
        FROM exercise
        WHERE id = %s AND day1 >= %s
        GROUP BY DATE_FORMAT(day1, '%%Y-%%m')
    """

    #current_date = datetime.now()
    #six_months_ago = current_date - timedelta(days=6*30)

# Generate a list of all the months in the last 6 months
    #months_range = pd.date_range(start=six_months_ago, end=current_date, freq='MS').strftime('%Y-%m').tolist()
    
# Execute the query for the last 6 months
    cursor.execute(sql_query_6_months1, (dynamic_id, six_months_ago))
    results_6_months1 = cursor.fetchall()
    
# Create a dictionary to store the results
    data_6_months1 = {month: 0 for month in months_range}
    #print(data_6_months)
    #print(results_6_months)
# Update data with results from the query
    for row in results_6_months1:
        month_year = row['month_year']
        data_6_months1[month_year] = int(row['total_minutes'])
    df2 = pd.DataFrame()
    
    df2['month_year']=data_6_months1.keys()
    df2['total_minutes']=data_6_months1.values()
    print(df2)
# Close cursor and connection


# Convert month_year to desired format
    df2['month_year'] = pd.to_datetime(df2['month_year']).dt.strftime('%b %Y')
#--------------------------right bicep curl----------------------------
    sql_query_6_months2 = """
            SELECT DATE_FORMAT(day1, '%%Y-%%m') AS month_year, SUM(r_curl_count) AS total_minutes
            FROM exercise
            WHERE id = %s AND day1 >= %s
            GROUP BY DATE_FORMAT(day1, '%%Y-%%m')
        """

    #current_date = datetime.now()
    #six_months_ago = current_date - timedelta(days=6*30)

# Generate a list of all the months in the last 6 months
    #months_range = pd.date_range(start=six_months_ago, end=current_date, freq='MS').strftime('%Y-%m').tolist()
    
# Execute the query for the last 6 months
    cursor.execute(sql_query_6_months2, (dynamic_id, six_months_ago))
    results_6_months2 = cursor.fetchall()
    
# Create a dictionary to store the results
    data_6_months2 = {month: 0 for month in months_range}
    #print(data_6_months)
    #print(results_6_months)
# Update data with results from the query
    for row in results_6_months2:
        month_year = row['month_year']
        data_6_months2[month_year] = int(row['total_minutes'])
    df3 = pd.DataFrame()
    
    df3['month_year']=data_6_months2.keys()
    df3['total_minutes']=data_6_months2.values()
    print(df3)
# Close cursor and connection


# Convert month_year to desired format
    df3['month_year'] = pd.to_datetime(df3['month_year']).dt.strftime('%b %Y')

#----------left bicep curl-----------------
    sql_query_6_months3 = """
            SELECT DATE_FORMAT(day1, '%%Y-%%m') AS month_year, SUM(l_curl_count) AS total_minutes
            FROM exercise
            WHERE id = %s AND day1 >= %s
            GROUP BY DATE_FORMAT(day1, '%%Y-%%m')
        """

    #current_date = datetime.now()
    #six_months_ago = current_date - timedelta(days=6*30)

# Generate a list of all the months in the last 6 months
    #months_range = pd.date_range(start=six_months_ago, end=current_date, freq='MS').strftime('%Y-%m').tolist()
    
# Execute the query for the last 6 months
    cursor.execute(sql_query_6_months3, (dynamic_id, six_months_ago))
    results_6_months3 = cursor.fetchall()
    
# Create a dictionary to store the results
    data_6_months3 = {month: 0 for month in months_range}
    #print(data_6_months)
    #print(results_6_months)
# Update data with results from the query
    for row in results_6_months3:
        month_year = row['month_year']
        data_6_months3[month_year] = int(row['total_minutes'])
    df4 = pd.DataFrame()
    
    df4['month_year']=data_6_months3.keys()
    df4['total_minutes']=data_6_months3.values()
    print(df4)
# Close cursor and connection


# Convert month_year to desired format
    df4['month_year'] = pd.to_datetime(df4['month_year']).dt.strftime('%b %Y')
    return render_template('trackprogress.html', a=df['month_year'].tolist(), b=df['total_minutes'].tolist(),c=df2['month_year'].tolist(), d=df2['total_minutes'].tolist(),e=df3['month_year'].tolist(), f=df3['total_minutes'].tolist(),g=df4['month_year'].tolist(), h=df4['total_minutes'].tolist())
#===============================================================================================================================================================
#+=======================================================================================================================================================================
#============================================================================================================================================================
@app.route('/weektrack')
def weekdiscussion():
    cursor = db.cursor()
    
    
    t = datetime.now().date()
    dynamic_id =session['user']
    '''seven_days_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')

        # Define the SQL query
    sql = """
    SELECT DATE_FORMAT(day1, '%%Y-%%m') AS month_year, SUM(plank_m) AS total_minutes
    FROM exercise
    WHERE id = %s AND day1 >= %s
    GROUP BY DATE_FORMAT(day1, '%%Y-%%m')
"""
        # Execute the query
    cursor.execute(sql, (dynamic_id, seven_days_ago))'''
    
    #seven_days_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    six_months_ago= (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
# Define the dates for the last 6 months
    #six_months_ago = (datetime.now() - timedelta(days=6*30)).strftime('%Y-%m-%d')

# Define the SQL query for the last 7 days
    
# Define the SQL query for the last 6 months
#----------plank 7 days-------------------
    sql_query_6_months ="""
    SELECT DATE_FORMAT(day1, '%%Y-%%m-%%d') AS day, COALESCE(SUM(plank_m), 0) AS total_minutes
    FROM exercise
    WHERE id = %s AND day1 >= %s
    GROUP BY DATE_FORMAT(day1, '%%Y-%%m-%%d')
"""

    current_date = datetime.now()
    six_months_ago = current_date - timedelta(days=6)

# Generate a list of all the months in the last 6 months
    months_range = pd.date_range(start=six_months_ago, end=current_date)

# Convert the date range to a list of strings in the format '%Y-%m-%d'
    months_range = months_range.strftime('%Y-%m-%d').tolist()
    
# Execute the query for the last 6 months
    cursor.execute(sql_query_6_months, (dynamic_id, six_months_ago))
    results_6_months = cursor.fetchall()
    
# Create a dictionary to store the results
    data_6_months = {month: 0 for month in months_range}
    #print(data_6_months)
    #print(results_6_months)
# Update data with results from the query
    for row in results_6_months:
        month_year = row['day']
        data_6_months[month_year] = int(row['total_minutes'])
    df = pd.DataFrame()
    print(months_range)
    print(results_6_months)
    print(df)
    df['month_year']=data_6_months.keys()
    df['total_minutes']=data_6_months.values()
    
# Close cursor and connection


# Convert month_year to desired format
    df['month_year'] = pd.to_datetime(df['month_year']).dt.strftime('%Y-%m-%d')
#---------------squats 7 days-----------------------
    sql_query_6_months1 ="""
    SELECT DATE_FORMAT(day1, '%%Y-%%m-%%d') AS day, COALESCE(SUM(squats_count), 0) AS total_minutes
    FROM exercise
    WHERE id = %s AND day1 >= %s
    GROUP BY DATE_FORMAT(day1, '%%Y-%%m-%%d')
"""

    current_date = datetime.now()
    six_months_ago = current_date - timedelta(days=6)

# Generate a list of all the months in the last 6 months
    months_range = pd.date_range(start=six_months_ago, end=current_date)

# Convert the date range to a list of strings in the format '%Y-%m-%d'
    months_range = months_range.strftime('%Y-%m-%d').tolist()
    
# Execute the query for the last 6 months
    cursor.execute(sql_query_6_months1, (dynamic_id, six_months_ago))
    results_6_months1 = cursor.fetchall()
    
# Create a dictionary to store the results
    data_6_months1 = {month: 0 for month in months_range}
    #print(data_6_months)
    #print(results_6_months)
# Update data with results from the query
    for row in results_6_months1:
        month_year = row['day']
        data_6_months1[month_year] = int(row['total_minutes'])
    df2 = pd.DataFrame()
    print(data_6_months1)
    print(results_6_months1)
    print(df)
    df2['month_year']=data_6_months1.keys()
    df2['total_minutes']=data_6_months1.values()
    
# Close cursor and connection


# Convert month_year to desired format
    df2['month_year'] = pd.to_datetime(df2['month_year']).dt.strftime('%Y-%m-%d')
#--------------------------right bicep curl----------------------------
    sql_query_6_months2 ="""
    SELECT DATE_FORMAT(day1, '%%Y-%%m-%%d') AS day, COALESCE(SUM(r_curl_count), 0) AS total_minutes
    FROM exercise
    WHERE id = %s AND day1 >= %s
    GROUP BY DATE_FORMAT(day1, '%%Y-%%m-%%d')
"""

    current_date = datetime.now()
    six_months_ago = current_date - timedelta(days=6)

# Generate a list of all the months in the last 6 months
    months_range = pd.date_range(start=six_months_ago, end=current_date)

# Convert the date range to a list of strings in the format '%Y-%m-%d'
    months_range = months_range.strftime('%Y-%m-%d').tolist()
    
# Execute the query for the last 6 months
    cursor.execute(sql_query_6_months2, (dynamic_id, six_months_ago))
    results_6_months2 = cursor.fetchall()
    
# Create a dictionary to store the results
    data_6_months2 = {month: 0 for month in months_range}
    #print(data_6_months)
    #print(results_6_months)
# Update data with results from the query
    for row in results_6_months2:
        month_year = row['day']
        data_6_months2[month_year] = int(row['total_minutes'])
    df3= pd.DataFrame()
    print(data_6_months2)
    print(results_6_months2)
    print(df3)
    df3['month_year']=data_6_months2.keys()
    df3['total_minutes']=data_6_months2.values()
    
# Close cursor and connection


# Convert month_year to desired format
    df3['month_year'] = pd.to_datetime(df3['month_year']).dt.strftime('%Y-%m-%d')
#----------left bicep curl-----------------
    sql_query_6_months3 ="""
    SELECT DATE_FORMAT(day1, '%%Y-%%m-%%d') AS day, COALESCE(SUM(l_curl_count), 0) AS total_minutes
    FROM exercise
    WHERE id = %s AND day1 >= %s
    GROUP BY DATE_FORMAT(day1, '%%Y-%%m-%%d')
"""

    current_date = datetime.now()
    six_months_ago = current_date - timedelta(days=6)

# Generate a list of all the months in the last 6 months
    months_range = pd.date_range(start=six_months_ago, end=current_date)

# Convert the date range to a list of strings in the format '%Y-%m-%d'
    months_range = months_range.strftime('%Y-%m-%d').tolist()
    
# Execute the query for the last 6 months
    cursor.execute(sql_query_6_months3, (dynamic_id, six_months_ago))
    results_6_months3 = cursor.fetchall()
    
# Create a dictionary to store the results
    data_6_months3 = {month: 0 for month in months_range}
    #print(data_6_months)
    #print(results_6_months)
# Update data with results from the query
    for row in results_6_months3:
        month_year = row['day']
        data_6_months3[month_year] = int(row['total_minutes'])
    df4 = pd.DataFrame()
    print(data_6_months3)
    print(results_6_months3)
    print(df4)
    df4['month_year']=data_6_months3.keys()
    df4['total_minutes']=data_6_months3.values()
    
# Close cursor and connection


# Convert month_year to desired format
    df4['month_year'] = pd.to_datetime(df4['month_year']).dt.strftime('%Y-%m-%d')
    return render_template('weekprogress.html', a=df['month_year'].tolist(), b=df['total_minutes'].tolist(),c=df2['month_year'].tolist(), d=df2['total_minutes'].tolist(),e=df3['month_year'].tolist(), f=df3['total_minutes'].tolist(),g=df4['month_year'].tolist(), h=df4['total_minutes'].tolist())

#================================================================================================================================================================
#==========================================================================================================================================================================
#===============================================================================================================================================================    

@app.route('/curl')
def curl():
    return render_template('curl.html')

@app.route('/squats')
def squats():
    return render_template('squats.html')

@app.route('/plank')
def plank():
    return render_template('plank.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace;boundary=frame')


@app.route('/video1')
def video1():
    return Response(generate_frames1(), mimetype='multipart/x-mixed-replace;boundary=frame')

@app.route('/video2')
def video2():
    return Response(generate_frames2(), mimetype='multipart/x-mixed-replace;boundary=frame')


@app.route('/start')
def start():
    start_camera()
    return 'Camera started!'

@app.route('/stop')
def stop():
    stop_camera()
    return 'Camera stopped!'

@app.route('/finish', methods=['GET','POST'])
def finish_stream():
    db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'QWE098*123asd',
    'database': 'hi',
    'cursorclass': pymysql.cursors.DictCursor
        }
    # Get today's date
    today= pd.to_datetime('today').date().strftime('%Y-%m-%d')
    print(today)
    #today = datetime.today().strftime('%d-%m-%Y')
    
    print(pm,ps,lc,rc,s)
    # Connect to the database
    connection = pymysql.connect(**db_config)
    current_username=session['user']
    
    try:
        with connection.cursor() as cursor:
            # Check if the entry exists for toda
            # y's date and the current user
            check_query = "SELECT * FROM exercise WHERE day1 = %s AND id = %s"
            cursor.execute(check_query, (today, current_username))
            existing_entry = cursor.fetchone()


            if existing_entry:
                # If entry exists, update the values
                update_query = "UPDATE exercise SET l_curl_count = l_curl_count + %s, r_curl_count = r_curl_count + %s,squats_count= squats_count+ %s, plank_m = plank_m + %s, plank_s = plank_s + %s WHERE day1 = %s AND id = %s"
                cursor.execute(update_query, (lc, rc, s, pm, ps, today, current_username))
            else:
                # If entry does not exist, insert a new row
                insert_query = "INSERT INTO exercise (id, day1, l_curl_count, r_curl_count, squats_count, plank_m, plank_s) VALUES (%s, %s, %s, %s, %s, %s, %s)"
                cursor.execute(insert_query, (current_username, today,lc, rc, s, pm, ps))
            connection.commit()
            lc =0
            rc=0
            s=0
            pm=0
            ps=0
    except:
        pass
        # Commit changes to the database
            

    return redirect('/page2')


if __name__ == '__main__':
    app.run(debug=True)
