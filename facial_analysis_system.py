"""
REAL-TIME FACIAL ANALYSIS SYSTEM
Author: Yodhitomo Sidhi Pranoto
Final Year Project - BSc Computer Science

This system performs real-time emotion detection, age estimation, and gender recognition
using a webcam. It combines a custom-trained emotion model (RAF-DB) with DeepFace for
age/gender detection.

Features:
- Real-time emotion detection (7 emotions)
- Age estimation
- Gender recognition
- Color-coded visual feedback
- Statistics tracking
- Screenshot capability
"""

import cv2
import numpy as np
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings('ignore')
import time
import threading
from collections import Counter

# DeepFace requires tf-keras when using TensorFlow 2.16+
# If it fails, age/gender detection is disabled but emotion still works
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except (ImportError, ValueError):
    DEEPFACE_AVAILABLE = False
    print("\u26a0\ufe0f  DeepFace not available. Age/gender detection disabled.")
    print("   Fix: pip install tf-keras")

# ========================================
# MAIN FACIAL ANALYSIS CLASS
# ========================================

class FacialAnalysisSystem:
    """
    Multi-feature facial analysis system
    Detects emotions, estimates age, and identifies gender in real-time
    """
    
    def __init__(self, emotion_model_path='best_emotion_model.h5'):
        """
        Initialize the facial analysis system
        
        Args:
            emotion_model_path: Path to your trained emotion detection model
        """
        print("="*70)
        print("FACIAL ANALYSIS SYSTEM")
        print("Real-Time Emotion, Age & Gender Detection")
        print("Author: Yodhitomo Sidhi Pranoto")
        print("="*70)
        print("\nInitializing system components...")
        
        # ===== LOAD EMOTION MODEL =====
        try:
            self.emotion_model = load_model(emotion_model_path)
            print(f"✓ Emotion model loaded from: {emotion_model_path}")
            print(f"  Input shape: {self.emotion_model.input_shape}")
        except:
            print(f"⚠ Could not load emotion model from {emotion_model_path}")
            print("  System will use DeepFace for all features")
            self.emotion_model = None
        
        # ===== EMOTION LABELS (RAF-DB ORDER) =====
        # These correspond to the 7 emotion classes
        self.emotion_labels = {
            0: 'Surprise',
            1: 'Fear',
            2: 'Disgust',
            3: 'Happiness',
            4: 'Sadness',
            5: 'Anger',
            6: 'Neutral'
        }
        
        # ===== COLOR CODING FOR EMOTIONS =====
        # Each emotion gets a unique color for visual feedback
        # Format: (Blue, Green, Red) in OpenCV
        self.emotion_colors = {
            'Surprise': (0, 255, 255),    # Yellow
            'Fear': (128, 0, 128),        # Purple
            'Disgust': (0, 128, 128),     # Teal
            'Happiness': (0, 255, 0),     # Green
            'Sadness': (255, 0, 0),       # Blue
            'Anger': (0, 0, 255),         # Red
            'Neutral': (255, 255, 255),   # White
            'happy': (0, 255, 0),         # For DeepFace compatibility
            'sad': (255, 0, 0),
            'angry': (0, 0, 255),
            'surprise': (0, 255, 255),
            'fear': (128, 0, 128),
            'disgust': (0, 128, 128),
            'neutral': (255, 255, 255)
        }
        
        # ===== FACE DETECTION SETUP =====
        # Haar Cascade - fast face detector from OpenCV
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # ===== STATISTICS TRACKING =====
        self.frame_count = 0            # Total frames processed
        self.detection_count = 0        # Total faces detected
        self.emotion_history = []       # Track recent emotions
        self.age_estimates = []         # Track age estimates

        # ===== BACKGROUND INFERENCE STATE =====
        # Runs emotion/age prediction on a separate thread so the
        # display loop never blocks waiting for the model.
        self._infer_lock = threading.Lock()
        self._latest_face = None        # Face ROI queued for inference
        self._infer_result = {}         # Latest result from inference thread
        self._infer_thread = threading.Thread(target=self._inference_worker, daemon=True)
        self._infer_thread.start()
        
        print("✓ Face detection initialized")
        print("✓ Color coding configured")
        print("✓ Statistics tracking ready")
        print("\n" + "="*70)
        print("SYSTEM READY!")
        print("="*70)
        print("\nControls:")
        print("  Q - Quit application")
        print("  S - Save screenshot")
        print("  R - Reset statistics")
        print("="*70 + "\n")
    
    # ========================================
    # BACKGROUND INFERENCE WORKER
    # ========================================

    def _inference_worker(self):
        """
        Runs continuously in a background thread.
        Picks up the latest queued face, runs the model + DeepFace,
        and stores the result.  The main loop never has to wait.
        """
        while True:
            face = None
            with self._infer_lock:
                if self._latest_face is not None:
                    face = self._latest_face.copy()
                    self._latest_face = None   # consume it

            if face is None:
                time.sleep(0.005)              # nothing to do – yield CPU
                continue

            result = {}

            # --- emotion model ---
            try:
                face_resized   = cv2.resize(face, (224, 224))
                face_norm      = face_resized.astype('float32') / 255.0
                face_input     = np.expand_dims(face_norm, axis=0)
                preds          = self.emotion_model.predict(face_input, verbose=0)[0]
                idx            = int(np.argmax(preds))
                result['emotion']    = self.emotion_labels[idx]
                result['confidence'] = float(preds[idx]) * 100
            except Exception:
                pass

            # --- age / gender via DeepFace ---
            if DEEPFACE_AVAILABLE:
                try:
                    analysis = DeepFace.analyze(
                        face,
                        actions=['age', 'gender'],
                        enforce_detection=False,
                        silent=True
                    )
                    if isinstance(analysis, list):
                        analysis = analysis[0]
                    result['age']    = analysis.get('age', 'N/A')
                    result['gender'] = analysis.get('dominant_gender', 'N/A')
                except Exception:
                    pass

            with self._infer_lock:
                self._infer_result = result

    # ========================================
    # EMOTION DETECTION METHOD
    # ========================================
    
    def detect_emotion_custom(self, face_img):
        """
        Detect emotion using custom trained model
        
        Args:
            face_img: Face region image (BGR format)
            
        Returns:
            emotion: Predicted emotion name
            confidence: Confidence score (0-1)
            all_probs: Probability for each emotion class
        """
        if self.emotion_model is None:
            return None, 0, None
        
        try:
            # Preprocess face for your model (MobileNetV2 expects 224x224)
            face_resized = cv2.resize(face_img, (224, 224))
            face_normalized = face_resized.astype('float32') / 255.0  # Normalize
            face_input = np.expand_dims(face_normalized, axis=0)  # Add batch dimension
            
            # Get predictions from your model
            predictions = self.emotion_model.predict(face_input, verbose=0)[0]
            
            # Get the emotion with highest probability
            emotion_idx = np.argmax(predictions)
            emotion = self.emotion_labels[emotion_idx]
            confidence = predictions[emotion_idx] * 100  # Convert to percentage
            
            return emotion, confidence, predictions
            
        except Exception as e:
            print(f"Error in emotion detection: {e}")
            return None, 0, None
    
    # ========================================
    # AGE & GENDER DETECTION METHOD
    # ========================================
    
    def detect_age_gender(self, face_img):
        """
        Detect age and gender using DeepFace library
        
        Args:
            face_img: Face region image
            
        Returns:
            age: Estimated age
            gender: Predicted gender (Male/Female)
        """
        if not DEEPFACE_AVAILABLE:
            return 'N/A', 'N/A'
        try:
            # Use DeepFace to analyze the face
            # This uses pre-trained models for age/gender
            analysis = DeepFace.analyze(
                face_img,
                actions=['age', 'gender'],   # What to detect
                enforce_detection=False,      # Don't fail if face not detected
                silent=True                   # Suppress DeepFace output
            )
            
            # DeepFace can return list or dict
            if isinstance(analysis, list):
                analysis = analysis[0]
            
            # Extract age and gender
            age = analysis.get('age', 'Unknown')
            gender = analysis.get('dominant_gender', 'Unknown')
            
            return age, gender
            
        except Exception as e:
            # Return N/A if detection fails
            return 'N/A', 'N/A'
    
    # ========================================
    # DRAW INFO PANEL METHOD
    # ========================================
    
    def draw_info_panel(self, frame):
        """
        Draw statistics panel on the frame
        
        Args:
            frame: Current video frame
            
        Returns:
            frame: Frame with info panel drawn
        """
        # ── Compact stats panel (top-left) ──
        # Smaller font (0.38) and tighter line spacing to keep it unobtrusive
        FONT      = cv2.FONT_HERSHEY_SIMPLEX
        FS        = 0.38   # font scale
        FT        = 1      # font thickness
        LINE_H    = 18     # pixels between lines
        PAD       = 8      # left/top padding inside box
        BOX_W     = 230
        BOX_H     = 105

        # Build the lines we want to show
        lines = [
            ("Facial Analysis System",  (0, 220, 220), 0.40, 1),
            (f"Frames: {self.frame_count}",    (200, 200, 200), FS, FT),
            (f"Faces:  {self.detection_count}", (200, 200, 200), FS, FT),
        ]
        if self.emotion_history:
            most_common = Counter(self.emotion_history).most_common(1)[0][0]
            lines.append((f"Trend:  {most_common}", (100, 220, 100), FS, FT))
        if self.age_estimates:
            avg_age = sum(self.age_estimates) / len(self.age_estimates)
            lines.append((f"Avg age:{avg_age:.0f}", (200, 200, 200), FS, FT))

        # Semi-transparent background (ROI only, no full-frame copy)
        roi   = frame[8 : 8+BOX_H, 8 : 8+BOX_W]
        black = np.zeros_like(roi)
        frame[8 : 8+BOX_H, 8 : 8+BOX_W] = cv2.addWeighted(roi, 0.25, black, 0.75, 0)

        # Draw each line
        for i, (txt, colour, fs, ft) in enumerate(lines):
            y = 8 + PAD + int((i + 1) * LINE_H * (fs / 0.38))
            cv2.putText(frame, txt, (8 + PAD, y), FONT, fs, colour, ft, cv2.LINE_AA)
        
        return frame
    
    # ========================================
    # MAIN DETECTION LOOP
    # ========================================
    
    def run(self):
        """
        Main loop - captures webcam and performs real-time analysis
        """
        # Open webcam – retry a few times because macOS sometimes needs a moment
        cap = None
        for attempt in range(5):
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                break
            print(f"  Camera not ready, retrying... ({attempt+1}/5)")
            time.sleep(1)

        if not cap or not cap.isOpened():
            print("❌ ERROR: Cannot open camera")
            print("Please check:")
            print("  1. Camera is connected")
            print("  2. No other app is using the camera")
            print("  3. Camera permissions are granted")
            return
        
        # Capture at 1280x720 for a wider landscape view
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Make the display window resizable and set a fixed size
        cv2.namedWindow('Facial Analysis - Yodhitomo Pranoto', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Facial Analysis - Yodhitomo Pranoto', 1280, 720)

        print("Camera opened successfully!")
        print("Starting real-time detection...\n")
        
        # For FPS calculation
        fps_start_time = time.time()
        fps_counter = 0
        current_fps = 0
        
        # Last known results (updated by background thread)
        last_emotion    = "Unknown"
        last_confidence = 0
        last_age        = "N/A"
        last_gender     = "N/A"

        # Face-detection runs every N frames to save CPU
        detect_every  = 2   # run HAAR every 2nd frame
        last_faces    = []  # reuse previous bboxes on skipped frames
        
        # ===== MAIN LOOP =====
        while True:
            # Read frame from webcam
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to grab frame")
                break
            
            # Update counters
            self.frame_count += 1
            fps_counter += 1
            
            # Calculate FPS every second
            if (time.time() - fps_start_time) > 1:
                current_fps = fps_counter / (time.time() - fps_start_time)
                fps_counter = 0
                fps_start_time = time.time()
            
            # ── Face detection (skip alternate frames to save CPU) ──
            if self.frame_count % detect_every == 0:
                gray       = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                last_faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(80, 80)   # slightly smaller min – still avoids noise
                )

            # ── Collect latest inference result from background thread ──
            with self._infer_lock:
                result = dict(self._infer_result)

            if result.get('emotion'):
                last_emotion    = result['emotion']
                last_confidence = result.get('confidence', 0)
                self.emotion_history.append(last_emotion)
                if len(self.emotion_history) > 100:
                    self.emotion_history.pop(0)
            if result.get('age'):
                last_age    = result['age']
                last_gender = result.get('gender', 'N/A')
                try:
                    self.age_estimates.append(int(last_age))
                    if len(self.age_estimates) > 50:
                        self.age_estimates.pop(0)
                except Exception:
                    pass

            # ── Draw detections and queue a face for the inference thread ──
            for (x, y, w, h) in last_faces:
                self.detection_count += 1
                face_roi = frame[y:y+h, x:x+w]

                # Hand the face to the background thread (non-blocking)
                with self._infer_lock:
                    self._latest_face = face_roi
                
                # Get color for current emotion
                color = self.emotion_colors.get(last_emotion, (255, 255, 255))
                
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                
                # ===== INFO BOX ABOVE/BELOW FACE =====
                info_box_height = 120
                info_y = y - info_box_height if y > info_box_height else y + h + 10
                
                # Semi-transparent background – use addWeighted on a ROI only
                # (much faster than copying the whole frame)
                x2 = min(x + w, frame.shape[1])
                y2 = min(info_y + info_box_height, frame.shape[0])
                y1 = max(info_y, 0)
                roi = frame[y1:y2, x:x2]
                black = np.zeros_like(roi)
                frame[y1:y2, x:x2] = cv2.addWeighted(roi, 0.35, black, 0.65, 0)
                
                text_y = info_y + 25
                cv2.putText(frame, f"Emotion: {last_emotion}", (x+5, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                text_y += 30
                if last_confidence > 0:
                    cv2.putText(frame, f"Confidence: {last_confidence:.1f}%", (x+5, text_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                text_y += 25
                cv2.putText(frame, f"Age: ~{last_age}", (x+5, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                text_y += 25
                cv2.putText(frame, f"Gender: {last_gender}", (x+5, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw statistics panel (skip the expensive full-frame copy inside)
            frame = self.draw_info_panel(frame)
            
            # Display FPS counter (top right)
            cv2.putText(frame, f"FPS: {current_fps:.1f}", 
                       (frame.shape[1] - 130, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show the frame
            cv2.imshow('Facial Analysis - Yodhitomo Pranoto', frame)
            
            # ===== HANDLE KEYBOARD INPUT =====
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                # Quit
                print("\n⏹ Shutting down system...")
                break
                
            elif key == ord('s') or key == ord('S'):
                # Save screenshot
                timestamp = int(time.time())
                filename = f'screenshot_{timestamp}.png'
                cv2.imwrite(filename, frame)
                print(f"📸 Screenshot saved: {filename}")
                
            elif key == ord('r') or key == ord('R'):
                # Reset statistics
                self.frame_count = 0
                self.detection_count = 0
                self.emotion_history = []
                self.age_estimates = []
                print("🔄 Statistics reset")
        
        # ===== CLEANUP =====
        cap.release()              # Release camera
        cv2.destroyAllWindows()    # Close all windows
        
        # ===== SHOW SESSION SUMMARY =====
        print("\n" + "="*70)
        print("SESSION SUMMARY")
        print("="*70)
        print(f"Total frames processed: {self.frame_count}")
        print(f"Total faces detected: {self.detection_count}")
        
        # Show emotion distribution
        if len(self.emotion_history) > 0:
            print("\nEmotion Distribution:")
            emotion_counts = Counter(self.emotion_history)
            for emotion, count in emotion_counts.most_common():
                percentage = (count / len(self.emotion_history)) * 100
                print(f"  {emotion}: {count} ({percentage:.1f}%)")
        
        # Show age statistics
        if len(self.age_estimates) > 0:
            avg_age = sum(self.age_estimates) / len(self.age_estimates)
            print(f"\nAverage age detected: {avg_age:.1f} years")
        
        print("="*70)
        print("\n✅ System stopped successfully")


# ========================================
# MAIN ENTRY POINT
# ========================================

def main():
    """
    Main function - creates and runs the system
    """
    import os

    model_folder = "models"

    # Check folder exists
    if not os.path.exists(model_folder):
        print(f"❌ Model folder '{model_folder}' not found.")
        return

    # Get all valid model files
    model_files = [
        f for f in os.listdir(model_folder)
        if f.endswith(('.h5', '.keras'))
    ]

    if not model_files:
        print("❌ No model files found in 'models/' folder.")
        return

    # Display options
    print("\nAvailable Emotion Models:")
    print("=" * 40)
    for i, model in enumerate(model_files):
        print(f"{i + 1}. {model}")

    # User selection
    while True:
        try:
            choice = int(input("\nSelect a model (number): "))
            if 1 <= choice <= len(model_files):
                selected_model = model_files[choice - 1]
                break
            else:
                print("⚠ Invalid selection. Try again.")
        except ValueError:
            print("⚠ Please enter a number.")

    # Full path
    model_path = os.path.join(model_folder, selected_model)

    print(f"\n✅ Using model: {selected_model}\n")

    try:
        system = FacialAnalysisSystem(emotion_model_path=model_path)
        system.run()

    except KeyboardInterrupt:
        print("\n\n⏹ System stopped by user (Ctrl+C)")

    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        print("Please check:")
        print("  1. Camera is accessible")
        print("  2. Required packages are installed (deepface, opencv-python, tensorflow)")
        print("  3. Emotion model file exists")


if __name__ == "__main__":
    main()
