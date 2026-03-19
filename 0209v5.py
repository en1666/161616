import cv2
import mediapipe as mp
import numpy as np
import screeninfo
import os
import time

# --- 設定區 ---
VIDEO_FILE = r'C:\Users\User\Desktop\media\0209v\5.雙手側舉0209.mp4' 
REPS_PER_LEG = 5  

should_exit = False
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils

def click_event(event, x, y, flags, param):
    global should_exit
    sw, sh = param
    if event == cv2.EVENT_LBUTTONDOWN:
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
    """
    修正後的裁切邏輯：保留影像頂部，確保側舉時能看到手部。
    """
    h, w = image.shape[:2]
    if (w/h) > (target_w/target_h):
        # 影像太寬，裁切兩邊
        new_w = int(h * (target_w/target_h))
        offset = (w - new_w) // 2
        crop = image[:, offset:offset+new_w]
    else:
        # 影像太長，裁切下方，保留上方
        new_h = int(w * (target_h/target_w))
        # 將 offset 設為 0，代表從最頂部開始取影像
        offset = 0 
        crop = image[offset:offset+new_h, :]
    return cv2.resize(crop, (target_w, target_h))

def draw_exit_button(img, sh):
    cv2.rectangle(img, (0, sh-150), (250, sh), (50, 50, 200), -1)
    cv2.putText(img, "TOUCH TO", (35, sh-90), 1, 2, (255, 255, 255), 2)
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

    # --- 1. 全螢幕播放示範 (修正播放延遲) ---
    demo_cap = cv2.VideoCapture(VIDEO_FILE)
    fps = demo_cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30
    wait_ms = int(1000 / fps)
    
    demo_frames = []
    while demo_cap.isOpened():
        ret, frame = demo_cap.read()
        if not ret or should_exit: break
        
        display_frame = letterbox_image(frame, sw, sh)
        draw_exit_button(display_frame, sh)
        cv2.imshow(win_name, display_frame)
        
        if cv2.waitKey(wait_ms) & 0xFF == 27: 
            should_exit = True
            break
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
            if current_side == "RIGHT":
                elbow = lm[mp_pose.PoseLandmark.LEFT_ELBOW] 
                shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
            else:
                elbow = lm[mp_pose.PoseLandmark.RIGHT_ELBOW]
                shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]

            if elbow.visibility > 0.7 and shoulder.visibility > 0.7:
                # 側舉：手肘上抬至肩膀高度
                if elbow.y < shoulder.y + 0.10: 
                    stage = "up"
                elif elbow.y > shoulder.y + 0.25 and stage == "up":
                    stage = "down"
                    if current_side == "RIGHT":
                        right_reps += 1
                        if right_reps >= REPS_PER_LEG:
                            current_side = "LEFT"
                            stage = "down"
                    else:
                        left_reps += 1

            is_correct = (elbow.y < shoulder.y + 0.10) if elbow.visibility > 0.7 else False
            color = (0, 255, 0) if is_correct else (0, 0, 255)
            mp_drawing.draw_landmarks(frame_display, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # UI 顯示
        cv2.rectangle(frame_display, (0, 0), (sw, 100), (30, 30, 30), -1)
        r_color = (0, 255, 255) if current_side == "RIGHT" else (255, 255, 255)
        l_color = (0, 255, 255) if current_side == "LEFT" else (255, 255, 255)
        
        cv2.putText(frame_display, f"RIGHT: {right_reps}/{REPS_PER_LEG}", (40, 65), 1, 2.5, r_color, 3)
        cv2.putText(frame_display, f"LEFT: {left_reps}/{REPS_PER_LEG}", (450, 65), 1, 2.5, l_color, 3)
        cv2.putText(frame_display, "LATERAL RAISE", (sw - 500, 65), 1, 2.8, (255, 255, 255), 3)
        draw_exit_button(frame_display, sh)

        # 順暢示範影片 (確保裁切保留頂部)
        elapsed = time.time() - start_time
        demo_idx = int(elapsed * fps) % len(demo_frames)
        thumb = crop_to_fill_top(demo_frames[demo_idx], 400, 300)
        frame_display[sh-320:sh-20, sw-420:sw-20] = thumb
        cv2.rectangle(frame_display, (sw-420, sh-320), (sw-20, sh-20), color, 5)

        cv2.imshow(win_name, frame_display)
        if cv2.waitKey(1) & 0xFF == 27 or (left_reps >= REPS_PER_LEG):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_trainer()