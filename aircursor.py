import cv2
import pyautogui
import numpy as np
from collections import deque
import argparse
import time

# Screen dimensions
screen_w, screen_h = pyautogui.size()

# Hand tracking parameters
prev_x, prev_y = 0, 0
smoothing = 2
click_threshold = 50  # pixels
click_history = deque(maxlen=5)

# Motion tracking
prev_gray = None
motion_buffer = deque(maxlen=10)  # Increased buffer size

# Scroll gestures
scroll_buffer = deque(maxlen=10)

# Click detection
def detect_click(fingers_up):
    click_history.append(fingers_up)
    
    # Left click (1 finger)
    if len(click_history) == 5 and all(f == 1 for f in click_history):
        pyautogui.click()
        click_history.clear()
        return "LEFT CLICK"
        
    # Right click (2 fingers)
    elif len(click_history) == 5 and all(f == 2 for f in click_history):
        pyautogui.rightClick()
        click_history.clear()
        return "RIGHT CLICK"
    
    return ""

# Hand detection using contours
def detect_hand(frame):
    # Convert to HSV and threshold skin color
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 30, 60), (20, 150, 255))
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Get largest contour (likely hand)
        hand_contour = max(contours, key=cv2.contourArea)
        return cv2.boundingRect(hand_contour)
    return None

# Add command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--fast', action='store_true', help='Max performance mode')
parser.add_argument('--debug', action='store_true', help='Debug mode with visuals')
parser.add_argument('--sensitivity', type=float, default=1.0, 
                   help='Gesture sensitivity (0.5-2.0)')
parser.add_argument('--terminal', action='store_true', help='Terminal-only mode')
args = parser.parse_args()

# Configure based on mode
if args.fast:
    resolution = (320, 240)
    skip_frames = 2
    show_video = False
elif args.debug:
    resolution = (640, 480) 
    skip_frames = 0
    show_video = True
elif args.terminal:
    resolution = None
    skip_frames = None
    show_video = False
else:  # balanced
    resolution = (480, 360)
    skip_frames = 1
    show_video = True

# Add deadzone threshold (adjust as needed)
DEADZONE_THRESHOLD = 0.05 # Lowered for more sensitivity

def process_frame(frame, resolution, prev_gray, motion_buffer, prev_x, prev_y):
    # Downsample and convert to grayscale
    small_frame = cv2.resize(frame, (resolution[0]//2, resolution[1]//2))
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate optical flow
    flow = None
    if prev_gray is not None:
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            pyr_scale=0.5,  # More sensitive to small motions
            levels=3,
            winsize=15,     # Larger window for stability
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
    
    return small_frame, gray, flow

def update_cursor_position(flow, motion_buffer, prev_x, prev_y):
    sensitivity = max(0.5, min(2.0, args.sensitivity))
    if flow is not None:
        motion_buffer.append((np.mean(flow[...,0]), np.mean(flow[...,1])))
        smooth_x = np.mean([m[0] for m in motion_buffer[-5:]])
        smooth_y = np.mean([m[1] for m in motion_buffer[-5:]])
        
        # Update cursor position with deadzone
        if abs(smooth_x) > DEADZONE_THRESHOLD or abs(smooth_y) > DEADZONE_THRESHOLD:
            cursor_x = prev_x + smooth_x * sensitivity * 20
            cursor_y = prev_y + smooth_y * sensitivity * 20
        else:
            cursor_x, cursor_y = prev_x, prev_y  # No movement below threshold
        
        pyautogui.moveTo(cursor_x, cursor_y)
        
        return cursor_x, cursor_y
    return prev_x, prev_y

def visualize_frame(frame, small_frame, gray, flow, prev_x, prev_y, hand_rect, fingers_up, action, drag_active):
    # Check and reset window positions if needed
    for win in ['Raw Camera Feed', 'Motion Magnitude', 'Processed Feed']:
        try:
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                cv2.namedWindow(win, cv2.WINDOW_NORMAL)
                cv2.moveWindow(win, 100 + 400 * len(cv2.getWindowImageRect(win)), 100)
        except:
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    
    # Display raw camera feed
    cv2.imshow('Raw Camera Feed', frame)
    cv2.moveWindow('Raw Camera Feed', 100, 100)
    
    # Visual feedback for motion
    if flow is not None:
        # Draw motion vectors
        h, w = frame.shape[:2]
        step = 16
        y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2,-1).astype(int)
        fx, fy = flow[y,x].T
        lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines + 0.5)
        cv2.polylines(frame, lines, 0, (0, 255, 0), thickness=1)

        # Display motion values
        smooth_x = np.mean([m[0] for m in motion_buffer[-5:]]) if motion_buffer else 0
        smooth_y = np.mean([m[1] for m in motion_buffer[-5:]]) if motion_buffer else 0
        cv2.putText(frame, f"Motion X: {smooth_x:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Motion Y: {smooth_y:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Display deadzone status
        is_deadzone_active = abs(smooth_x) <= DEADZONE_THRESHOLD and abs(smooth_y) <= DEADZONE_THRESHOLD
        status_text = "Status: DEADZONE" if is_deadzone_active else "Status: ACTIVE"
        color = (0, 0, 255) if is_deadzone_active else (0, 255, 0)
        cv2.putText(frame, status_text, (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    magnitude = np.sqrt(flow[...,0]**2 + flow[...,1]**2) if flow is not None else np.zeros_like(gray)
    # Ensure the Motion Magnitude window is always shown for diagnostics
    max_magnitude = np.max(magnitude)
    display_magnitude = magnitude / max_magnitude if max_magnitude > 0 else magnitude
    cv2.imshow('Motion Magnitude', display_magnitude)
    cv2.moveWindow('Motion Magnitude', 100, 600)
    
    # Display processed frame
    cv2.putText(frame, f"Cursor: ({int(prev_x)},{int(prev_y)})", 
                (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.putText(frame, f"Sensitivity: {max(0.5, min(2.0, args.sensitivity)):.1f}", 
                (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
    
    status = "CLICK" if len(click_history) == 5 else "SCROLL" if len(scroll_buffer) >= 5 else "MOVE"
    cv2.putText(frame, status, (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    
    if hand_rect:
        cv2.rectangle(frame, (hand_rect[0]*2, hand_rect[1]*2), (hand_rect[0]*2+hand_rect[2]*2, hand_rect[1]*2+hand_rect[3]*2), (0, 255, 0), 2)
        cv2.putText(frame, f"Fingers: {fingers_up}", (hand_rect[0]*2, hand_rect[1]*2-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        if action:
            cv2.putText(frame, action, (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    if drag_active:
        cv2.putText(frame, "DRAG", (50, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    
    cv2.imshow('Processed Feed', cv2.resize(frame, (resolution[0], resolution[1])))
    cv2.moveWindow('Processed Feed', 800, 100)

def calibrate_sensitivity():
    print("Calibrating motion sensitivity...")
    cap = cv2.VideoCapture(0)
    motions = []
    prev_gray = None
    
    for _ in range(30):  # Sample 30 frames
        ret, frame = cap.read()
        if not ret: continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            motions.append(np.mean(np.abs(flow)))
        prev_gray = gray
        
    avg_motion = np.mean(motions)
    print(f"Calibration complete - Avg motion: {avg_motion:.2f}")
    return max(0.5, min(2.0, 1.0 / avg_motion))  # Normalized sensitivity

def initialize_camera():
    """Only use known working configuration (index 0)"""
    cap = cv2.VideoCapture(0)  # Only index that works
    
    if not cap.isOpened():
        print("CRITICAL ERROR: Camera not accessible")
        print("Please:")
        print("1. Check macOS Privacy Settings > Camera")
        print("2. Restart computer")
        print("3. Try external USB camera")
        return None
    
    # Set conservative resolution for reliability
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    return cap

def test_camera():
    """Standalone camera test with visual feedback"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera failed to open")
        return
    
    print("Camera test running - Press ESC to exit")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame read error")
            break
            
        cv2.imshow('Camera Test', frame)
        if cv2.waitKey(1) == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Camera test complete")

# Main loop
def main():
    global prev_gray, prev_x, prev_y # Ensure globals are accessible
    cap = initialize_camera()
    if not cap:
        return

    # Reset state
    prev_gray = None
    prev_x, prev_y = pyautogui.position()
    frame_counter = 0

    # Calibrate if not in terminal mode
    if not args.terminal:
        # args.sensitivity = calibrate_sensitivity()
        pass # Skipping calibration for now

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Frame read error, exiting...")
            break

        # Process frame for motion
        small_frame, gray, flow = process_frame(frame, resolution, prev_gray, motion_buffer, prev_x, prev_y)

        # Update cursor position based on motion
        prev_x, prev_y = update_cursor_position(flow, motion_buffer, prev_x, prev_y)

        # Show visuals if not in fast/terminal mode
        if show_video:
            visualize_frame(frame, small_frame, gray, flow, prev_x, prev_y, None, 0, "", False)
        
        prev_gray = gray

        # Exit on ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break
            
    cap.release()
    cv2.destroyAllWindows()
    print("\nExited cleanly.")

if __name__ == "__main__":
    main()
