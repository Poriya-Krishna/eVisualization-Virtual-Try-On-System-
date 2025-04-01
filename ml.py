import cv2
import dlib
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Load pre-trained dlib models
try:
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor.dat")  # Ensure file exists rajdeep thorat was here
except RuntimeError:
    raise FileNotFoundError("Missing required 'shape_predictor.dat' file.")

# Load multiple accessory images (spectacles) with transparency
accessory_paths = [ "rectangle_frame4.png",  
                  "rectungalar_frame1.png", "rectungular_frame2.png",  "round_frame1.png", 
                  "round_frame2.png", "round_frame3.png", "round_frame4.png", "rimeless_glass3.png", "rimeless_glasses1.png", 
                   "rimeless_glasses5.png", "rimeless_glasses6.png", "cateyes1.png", "cateyes2.png", 
                  "rimeless_glasses4.png",  "wider_frame1.png", "wider_frame2.png"]
accessories = []

# Load each accessory and keep track of which ones loaded successfully
loaded_paths = []
for path in accessory_paths:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is not None:
        accessories.append(img)
        loaded_paths.append(path)
    else:
        print(f"Warning: Could not load {path}")

if not accessories:
    raise FileNotFoundError("No valid accessory images found.")

# Update accessory_paths to only include successfully loaded images
accessory_paths = loaded_paths

# Detailed Glasses Recommendations - make sure we only reference valid paths
GLASSES_RECOMMENDATIONS = {
    "Round": {
        "frames": ["rectangle_frame4.png", "rectungalar_frame1.png", "rectungular_frame2.png"],
        "description": "Rectangular frames to add angles to your round face"
    },
    "Oval": {
        "frames": ["glasses1.png", "glasses2.png", "glasses3.png", "rectangle_frame4.png", "rectangular_frame3.png", 
                  "rectungalar_frame1.png", "rectungular_frame2.png", "rimeless_glasses5.png", "round_frame1.png", 
                  "round_frame2.png", "round_frame3.png", "round_frame4.png", "rimeless_glass3.png", "rimeless_glasses1.png", 
                  "rimeless_glasses4.png", "rimeless_glasses5.png", "rimeless_glasses6.png", "cateyes1.png", "cateyes2.png", 
                  "wider_frame1.png", "wider_frame2.png"],
        "description": "Almost any frame style works well with oval faces"
    },
    "Square": {
        "frames": ["rimeless_glasses5.png", "round_frame1.png", "round_frame2.png", "round_frame3.png", "round_frame4.png"],
        "description": "Round frames to soften strong angular features"
    },
    "Heart": {
        "frames": ["rimeless_glass3.png", "rimeless_glasses1.png", "rimeless_glasses4.png", "rimeless_glasses5.png", "rimeless_glasses6.png"],
        "description": "Lighter frames to balance broad forehead"
    },
    "Diamond": {
        "frames": ["cateyes1.png", "cateyes2.png", "round_frame2.png", "rimeless_glasses6.png", "rimeless_glasses5.png", "rimeless_glasses4.png"],
        "description": "Cat-eye frames to highlight cheekbones"
    },
    "Oblong": {
        "frames": ["wider_frame1.png", "wider_frame2.png"],
        "description": "Wider frames to create visual balance"
    }
}

# Filter recommendations to only include successfully loaded images
for face_shape, data in GLASSES_RECOMMENDATIONS.items():
    filtered_frames = [frame for frame in data["frames"] if frame in accessory_paths]
    # If all frames were filtered out, add a default
    if not filtered_frames and accessories:
        filtered_frames = [accessory_paths[0]]
    GLASSES_RECOMMENDATIONS[face_shape]["frames"] = filtered_frames

# Selected accessory index
selected_accessory_index = 0  # Default to first accessory

# For recommendation traversal
current_recommendation_index = 0
recommendation_frames = []

# Function to rotate image
def rotate_image(image, angle):
    """
    Rotates an image around its center by the given angle.
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)
    rotated = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    return rotated

# Function to overlay image
def overlay_image(background, overlay, x, y):
    """
    Overlays a transparent PNG onto a background image at (x, y) using alpha blending.
    Ensures overlay stays within frame bounds.
    """
    h, w = overlay.shape[:2]
    bh, bw = background.shape[:2]
    
    # Adjust if overlay goes beyond frame boundaries
    if x < 0:
        overlay = overlay[:, -x:]
        w = overlay.shape[1]
        x = 0
    if y < 0:
        overlay = overlay[-y:, :]
        h = overlay.shape[0]
        y = 0
    if x + w > bw:
        overlay = overlay[:, :bw - x]
        w = overlay.shape[1]
    if y + h > bh:
        overlay = overlay[:bh - y, :]
        h = overlay.shape[0]
    
    if overlay.shape[0] == 0 or overlay.shape[1] == 0:
        return background  # Skip overlay if nothing remains
    
    alpha_s = overlay[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(3):
        background[y:y+h, x:x+w, c] = (alpha_s * overlay[:, :, c] + alpha_l * background[y:y+h, x:x+w, c])
    
    return background

# Function to overlay accessory
def overlay_accessory(image, accessory, landmarks, left_eye_idx, right_eye_idx):
    """
    Overlays a selected accessory (e.g., spectacles) based on eye positions.
    """
    # Get exact eye positions tamdi chamdi ramesh
    left_eye = np.mean(landmarks[list(left_eye_idx)], axis=0).astype(int)
    right_eye = np.mean(landmarks[list(right_eye_idx)], axis=0).astype(int)
    
    # Get eye distance and face width for proper scaling
    eye_distance = np.linalg.norm(right_eye - left_eye)
    
    # Calculate appropriate scale for glasses
    glasses_width = int(eye_distance * 2.2)  # Make glasses a bit wider than eye distance
    glasses_height = int(glasses_width * accessory.shape[0] / accessory.shape[1])
    
    # Resize accessory
    resized_accessory = cv2.resize(accessory, (glasses_width, glasses_height), interpolation=cv2.INTER_AREA)
    
    # Calculate angle between eyes for rotation
    angle = np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))
    rotated_accessory = rotate_image(resized_accessory, angle)
    
    # Calculate center point between eyes
    eye_center_x = (left_eye[0] + right_eye[0]) // 2
    eye_center_y = (left_eye[1] + right_eye[1]) // 2
    
    # Position glasses precisely relative to eye center
    x = eye_center_x - (rotated_accessory.shape[1] // 2)
    y = int(eye_center_y - (rotated_accessory.shape[0] * 0.5))
    
    # Apply the overlay
    return overlay_image(image, rotated_accessory, x, y)

# Function to calculate face shape
def calculate_face_shape(landmarks):
    """
    Advanced face shape detection using multiple geometric measurements.
    """
    # Key facial points
    left_face = landmarks[0]
    right_face = landmarks[16]
    left_cheekbone = landmarks[2]
    right_cheekbone = landmarks[14]
    forehead_left = landmarks[17]
    forehead_right = landmarks[26]
    chin = landmarks[8]

    # Precise measurements
    face_width = np.linalg.norm(left_face - right_face)
    face_height = np.linalg.norm(np.mean([landmarks[19], landmarks[24]], axis=0) - chin)
    
    forehead_width = np.linalg.norm(forehead_left - forehead_right)
    cheekbone_width = np.linalg.norm(left_cheekbone - right_cheekbone)
    
    # Ratio calculations
    width_height_ratio = face_width / face_height

    # Enhanced shape detection logic
    if 0.9 <= width_height_ratio <= 1.1:
        if forehead_width > cheekbone_width * 1.1:
            return "Heart"
        elif cheekbone_width > face_width * 0.95:
            return "Square"
        else:
            return "Round"
    
    elif width_height_ratio < 0.85:
        return "Oblong"
    
    elif width_height_ratio > 1.2:
        return "Diamond" if cheekbone_width > face_width * 0.9 else "Oval"
    
    return "Oval"

# Function to recommend glasses
def recommend_glasses(face_shape):
    """
    Recommend glasses based on face shape with additional context.
    Returns dict with recommended frames and current index
    """
    global recommendation_frames, current_recommendation_index
    
    # Get recommendation for this face shape
    recommendations = GLASSES_RECOMMENDATIONS.get(face_shape, 
        {"frames": [accessory_paths[0]], "description": "Versatile frames"})
    
    # Store all recommended frames for traversal
    recommendation_frames = recommendations["frames"]
    current_recommendation_index = 0
    
    if not recommendation_frames:
        # If no frames are recommended (perhaps they didn't load), use the first available accessory
        if accessories:
            return {
                "index": 0,
                "description": recommendations["description"],
                "total_frames": 1,
                "current_frame": 1
            }
    
    # Find index of first recommended frame
    frame_path = recommendation_frames[current_recommendation_index]
    for i, path in enumerate(accessory_paths):
        if path == frame_path:
            return {
                "index": i,
                "description": recommendations["description"],
                "total_frames": len(recommendation_frames),
                "current_frame": 1,
                "frame_name": frame_path
            }
    
    # Fallback to first accessory
    return {
        "index": 0,
        "description": recommendations["description"],
        "total_frames": len(recommendation_frames),
        "current_frame": 1,
        "frame_name": accessory_paths[0]
    }

# Function to get next recommended frame
def get_next_recommended_frame(face_shape):
    """
    Get the next recommended frame for the current face shape
    """
    global current_recommendation_index, recommendation_frames
    
    # Make sure we have recommendations for this face shape
    if not recommendation_frames:
        recommendation_frames = GLASSES_RECOMMENDATIONS.get(face_shape, {"frames": [accessory_paths[0]]})["frames"]
    
    if not recommendation_frames:
        # If still no frames, use the first available accessory
        return {
            "index": 0,
            "description": GLASSES_RECOMMENDATIONS.get(face_shape, {"description": "Versatile frames"})["description"],
            "total_frames": 1,
            "current_frame": 1,
            "frame_name": accessory_paths[0] if accessory_paths else "Unknown"
        }
    
    # Move to next frame in recommendations
    current_recommendation_index = (current_recommendation_index + 1) % len(recommendation_frames)
    
    # Get the path of the next recommended frame
    frame_path = recommendation_frames[current_recommendation_index]
    
    # Find index of this frame in accessories
    for i, path in enumerate(accessory_paths):
        if path == frame_path:
            return {
                "index": i,
                "description": GLASSES_RECOMMENDATIONS[face_shape]["description"],
                "total_frames": len(recommendation_frames),
                "current_frame": current_recommendation_index + 1,
                "frame_name": frame_path
            }
    
    # Fallback if frame not found
    return {
        "index": 0,
        "description": GLASSES_RECOMMENDATIONS[face_shape]["description"],
        "total_frames": len(recommendation_frames),
        "current_frame": current_recommendation_index + 1,
        "frame_name": accessory_paths[0] if accessory_paths else "Unknown"
    }

# New machine learning function with accuracy visualization
def create_ml_face_shape_model(landmarks_data, labels):
    """
    Create a machine learning model to predict face shapes with accuracy visualization
    """
    # Flatten landmarks for ML processing
    X = np.array([landmark.flatten() for landmark in landmarks_data])
    y = np.array(labels)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest Classifier with multiple estimators
    n_estimators_range = [10, 50, 100, 200, 300]
    train_scores = []
    test_scores = []

    for n_estimators in n_estimators_range:
        rf_classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        rf_classifier.fit(X_train_scaled, y_train)
        
        # Calculate and store scores
        train_scores.append(rf_classifier.score(X_train_scaled, y_train))
        test_scores.append(rf_classifier.score(X_test_scaled, y_test))

    # Predict and generate visualization
    final_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    final_classifier.fit(X_train_scaled, y_train)
    y_pred = final_classifier.predict(X_test_scaled)

    # Confusion Matrix Visualization
    plt.figure(figsize=(15, 5))
    
    # Subplot 1: Confusion Matrix
    plt.subplot(1, 2, 1)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=np.unique(labels), 
                yticklabels=np.unique(labels))
    plt.title('Face Shape Prediction Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # Subplot 2: Accuracy vs Number of Estimators
    plt.subplot(1, 2, 2)
    plt.plot(n_estimators_range, train_scores, label='Training Accuracy', marker='o')
    plt.plot(n_estimators_range, test_scores, label='Testing Accuracy', marker='o')
    plt.title('Model Accuracy vs Number of Estimators')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Classification Report
    print(classification_report(y_test, y_pred))

    return final_classifier, scaler

# Initialize webcam
cap = cv2.VideoCapture(0)
# Set larger camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

left_eye_idx, right_eye_idx = range(36, 42), range(42, 48)

# For ensuring consistent detection
face_detected = False
last_valid_landmarks = None
last_face_shape = "Oval"  # Default face shape
last_recommendation = None

# ML Model Training Data
landmarks_collection = []
labels_collection = []
ml_model = None
ml_scaler = None

# Face shape distribution tracking
face_shape_distribution = {
    "Round": 0,
    "Oval": 0,
    "Square": 0,
    "Heart": 0,
    "Diamond": 0,
    "Oblong": 0
}

# Main loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    if faces:
        face_detected = True
        for face in faces:
            # Get facial landmarks
            landmarks_obj = predictor(gray, face)
            landmarks = np.array([[p.x, p.y] for p in landmarks_obj.parts()])
            
            # Calculate face shape
            try:
                face_shape = calculate_face_shape(landmarks)
                last_face_shape = face_shape
                
                # Increment face shape distribution counter
                face_shape_distribution[face_shape] += 1
                
                # Collect landmarks for ML training
                landmarks_collection.append(landmarks)
                labels_collection.append(face_shape)

                # Limit collection to prevent memory issues
                if len(landmarks_collection) > 100:
                    landmarks_collection.pop(0)
                    labels_collection.pop(0)
            except Exception as e:
                # Fallback to last known shape if calculation fails
                face_shape = last_face_shape
            
            # Overlay accessory
            last_valid_landmarks = landmarks
            frame = overlay_accessory(frame, accessories[selected_accessory_index], landmarks, left_eye_idx, right_eye_idx)
    
    elif face_detected and last_valid_landmarks is not None:
        # If face was previously detected but not in current frame
        frame = overlay_accessory(frame, accessories[selected_accessory_index], last_valid_landmarks, left_eye_idx, right_eye_idx)
    
    # Display face shape and recommendation
    cv2.putText(frame, f"Face Shape: {last_face_shape}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    if last_recommendation:
        cv2.putText(frame, 
            f"Recommendation: {last_recommendation['description']}", 
            (10, 60), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Show current frame and total frames
        cv2.putText(frame, 
            f"Frame {last_recommendation.get('current_frame', 1)} of {last_recommendation.get('total_frames', 1)}: {last_recommendation.get('frame_name', '')}", 
            (10, 90), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    # Display controls
    cv2.putText(frame, "Press 'n': Next  'p': Previous  'r': Recommend  't': Next Recommendation  'm': ML Model  'g': Graph  'q': Quit", 
                (10, frame.shape[0] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imshow("Virtual Try-On", frame)
    key = cv2.waitKey(1) & 0xFF
    
    # Key handling
    if key == ord("q"):
        break
    elif key == ord("n"):  # Press 'n' to switch to the next glasses
        selected_accessory_index = (selected_accessory_index + 1) % len(accessories)
    elif key == ord("p"):  # Press 'p' to switch to the previous glasses
        selected_accessory_index = (selected_accessory_index - 1) % len(accessories)
    elif key == ord("r") and last_face_shape:  # Press 'r' for recommendation
        last_recommendation = recommend_glasses(last_face_shape)
        selected_accessory_index = last_recommendation["index"]
    elif key == ord("t") and last_face_shape:  # Press 't' for next recommended frame
        last_recommendation = get_next_recommended_frame(last_face_shape)
        selected_accessory_index = last_recommendation["index"]
    elif key == ord("m"):  # Press 'm' for Machine Learning Analysis
        if len(landmarks_collection) > 10:  # Ensure we have enough data
            ml_model, ml_scaler = create_ml_face_shape_model(landmarks_collection, labels_collection)
            print("Machine Learning Model Trained Successfully!")
        else:
            print("Not enough data for ML model. Keep detecting faces.")
    elif key == ord("g"):  # Press 'g' to generate face shape distribution graph
        plt.figure(figsize=(10, 6))
        plt.bar(face_shape_distribution.keys(), face_shape_distribution.values())
        plt.title("Face Shape Distribution")
        plt.xlabel("Face Shapes")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()

cap.release()
cv2.destroyAllWindows()