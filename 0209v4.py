import cv2
import mediapipe as mp
import numpy as np
import screeninfo
import os
import time

# --- 設定區 ---
VIDEO_FILE = r'C:\Users\User\Desktop\media\0209v\4.高抬腿0209.mp4' 
REPS_PER_LEG = 5  

should_exit = False
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils

def click_event(event, x, y, flags, param):
    global should_exit
    sw, sh = param
    if event == cv2.EVENT_LBUTTONDOWN:
        # 點擊左下角區域 (寬250, 高150)
        if 0 < x < 250 and (sh - 150) < y < sh:
            should_exit = True

def letterbox_image(image, target_w, target_h):
    h, w = image.shape[:2]
    aspect = w / h
    target_aspect = target_w / target_h
    if target_aspect > aspect:
        new_h = target_h
        new_w = int(aspect * new_h)
    else:
        new_w = target_w
        new_h = int(new_w / aspect)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y_off, x_off = (target_h-new_h)//2, (target_w-new_w)//2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
    return canvas

def crop_to_fill_top(image, target_w, target_h):
    h, w = image.shape[:2]
    if (w/h) > (target_w/target_h):
        new_w = int(h * (target_w/target_h))
        offset = (w - new_w) // 2
        crop = image[:, offset:offset+new_w]
    else:
        new_h = int(w * (target_h/target_w))
        offset = int((h - new_h) * 0.45) 
        crop = image[offset:offset+new_h, :]
    return cv2.resize(crop, (target_w, target_h))

def draw_exit_button(img, sh):
    """繪製兩行文字的離開按鈕"""
    # 紅色底框
    cv2.rectangle(img, (0, sh-150), (250, sh), (50, 50, 200), -1)
    # 第一行文字 (TOUCH TO)
    cv2.putText(img, "TOUCH TO", (35, sh-90), 1, 2, (255, 255, 255), 2)
    # 第二行文字 (EXIT)
    cv2.putText(img, "EXIT", (35, sh-40), 1, 2, (255, 255, 255), 2)

def run_trainer():
    global should_exit
    try:
        monitors = screeninfo.get_monitors()
        sw, sh = monitors[0].width, monitors[0].height
    except:
        sw, sh = 1280, 720

    if not os.path.exists(VIDEO_FILE):
        print(f"找不到檔案: {VIDEO_FILE}")
        return

    win_name = 'AI Trainer'
    cv2.namedWindow(win_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback(win_name, click_event, param=(sw, sh))

    # --- 1. 全螢幕播放示範 ---
    demo_cap = cv2.VideoCapture(VIDEO_FILE)
    fps = demo_cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30
    demo_data, demo_frames = [], []
    
    while demo_cap.isOpened():
        ret, frame = demo_cap.read()
        if not ret or should_exit: break
        
        display_frame = letterbox_image(frame, sw, sh)
        draw_exit_button(display_frame, sh) # 呼叫新設計的按鈕
        
        cv2.imshow(win_name, display_frame)
        if cv2.waitKey(1) & 0xFF == 27: 
            should_exit = True
            break
        
        res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            ref_y = (lm[mp_pose.PoseLandmark.LEFT_KNEE].y + lm[mp_pose.PoseLandmark.RIGHT_KNEE].y) / 2
            demo_data.append(ref_y)
            demo_frames.append(frame)
    demo_cap.release()

    if should_exit: return

    # --- 2. 實時計次 ---
    cap = cv2.VideoCapture(0)
    left_reps, right_reps = 0, 0
    current_side = "RIGHT" 
    stage = "down"
    start_time = time.time()

    while cap.isOpened() and not should_exit:
        success, frame = cap.read()
        if not success: break
        
        frame = cv2.flip(frame, 1) 
        frame_display = frame.copy()
        frame_display = letterbox_image(frame_display, sw, sh)
        
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        color = (0, 0, 255)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            
            # 鏡像邏輯對應 (右腳做完換左腳)
            if current_side == "RIGHT":
                knee = lm[mp_pose.PoseLandmark.LEFT_KNEE] 
                hip = lm[mp_pose.PoseLandmark.LEFT_HIP]
            else:
                knee = lm[mp_pose.PoseLandmark.RIGHT_KNEE]
                hip = lm[mp_pose.PoseLandmark.RIGHT_HIP]

            if knee.visibility > 0.7 and hip.visibility > 0.7:
                if knee.y < hip.y + 0.12:
                    stage = "up"
                elif knee.y > hip.y + 0.22 and stage == "up":
                    stage = "down"
                    if current_side == "RIGHT":
                        right_reps += 1
                        if right_reps >= REPS_PER_LEG:
                            current_side = "LEFT"
                            stage = "down"
                    else:
                        left_reps += 1

            is_correct = (knee.y < hip.y + 0.12) if knee.visibility > 0.7 else False
            color = (0, 255, 0) if is_correct else (0, 0, 255)
            mp_drawing.draw_landmarks(frame_display, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # --- UI 繪製 ---
        cv2.rectangle(frame_display, (0, 0), (sw, 100), (30, 30, 30), -1)
        r_color = (0, 255, 255) if current_side == "RIGHT" else (255, 255, 255)
        l_color = (0, 255, 255) if current_side == "LEFT" else (255, 255, 255)
        
        cv2.putText(frame_display, f"RIGHT: {right_reps}/{REPS_PER_LEG}", (40, 65), 1, 2.5, r_color, 3)
        cv2.putText(frame_display, f"LEFT: {left_reps}/{REPS_PER_LEG}", (450, 65), 1, 2.5, l_color, 3)
        cv2.putText(frame_display, "HIGH KNEES", (sw - 400, 65), 1, 2.8, (255, 255, 255), 3)

        # 呼叫雙行離開按鈕
        draw_exit_button(frame_display, sh)

        # --- 順暢示範影片 ---
        elapsed = time.time() - start_time
        demo_idx = int(elapsed * fps) % len(demo_frames)
        
        thumb_w, thumb_h = 400, 300
        thumb = crop_to_fill_top(demo_frames[demo_idx], thumb_w, thumb_h)
        frame_display[sh-thumb_h-20:sh-20, sw-thumb_w-20:sw-20] = thumb
        cv2.rectangle(frame_display, (sw-thumb_w-20, sh-thumb_h-20), (sw-20, sh-20), color, 5)

        cv2.imshow(win_name, frame_display)
        
        if cv2.waitKey(1) & 0xFF == 27 or (left_reps >= REPS_PER_LEG):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_trainer()