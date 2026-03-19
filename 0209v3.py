import cv2
import mediapipe as mp
import numpy as np
import screeninfo
import os

# --- 設定區 ---
VIDEO_FILE = r'C:\Users\User\Desktop\media\0209v\3.肩膀繞圈0209.mp4' 
TARGET_REPS = 10 

# 全域變數
should_exit = False

# 使用 MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
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
    h, w = image.shape[:2]
    if (w/h) > (target_w/target_h):
        new_w = int(h * (target_w/target_h))
        offset = (w - new_w) // 2
        crop = image[:, offset:offset+new_w]
    else:
        new_h = int(w * (target_h/target_w))
        offset = int((h - new_h) * 0.30) 
        crop = image[offset:offset+new_h, :]
    return cv2.resize(crop, (target_w, target_h))

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
    demo_data, demo_frames = [], []
    while demo_cap.isOpened():
        ret, frame = demo_cap.read()
        if not ret or should_exit: break
        
        display_frame = letterbox_image(frame, sw, sh)
        cv2.rectangle(display_frame, (0, sh-150), (250, sh), (50, 50, 200), -1)
        cv2.putText(display_frame, "TOUCH TO", (30, sh-90), 1, 2, (255, 255, 255), 2)
        cv2.putText(display_frame, "EXIT", (30, sh-40), 1, 2, (255, 255, 255), 2)
        
        cv2.imshow(win_name, display_frame)
        if cv2.waitKey(20) & 0xFF == 27: 
            should_exit = True
            break
        
        res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            avg_y = (lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y + lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y) / 2
            demo_data.append(avg_y)
            demo_frames.append(frame)
    demo_cap.release()

    if should_exit: return

    # --- 2. 實時計次 ---
    cap = cv2.VideoCapture(0)
    reps = 0
    stage = "down" 
    demo_idx = 0

    while cap.isOpened() and not should_exit:
        success, frame = cap.read()
        if not success: break
        
        frame = cv2.flip(frame, 1)
        frame = letterbox_image(frame, sw, sh)
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        color = (0, 0, 255)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            sh_l, sh_r = lm[mp_pose.PoseLandmark.LEFT_SHOULDER], lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            hip_l, hip_r = lm[mp_pose.PoseLandmark.LEFT_HIP], lm[mp_pose.PoseLandmark.RIGHT_HIP]
            
            current_avg_sh_y = (sh_l.y + sh_r.y) / 2
            current_avg_hip_y = (hip_l.y + hip_r.y) / 2

            # 優化 3: 放寬比對示範影片的容錯率 (0.12 -> 0.18)
            is_correct = abs(current_avg_sh_y - demo_data[demo_idx]) < 0.18
            color = (0, 255, 0) if is_correct else (0, 0, 255)

            # 計次邏輯：計算肩膀與髖部的相對垂直距離
            rel_height = current_avg_hip_y - current_avg_sh_y
            
            # 優化 1 & 2: 降低計次門檻
            if rel_height > 0.41: # 聳肩判定線 (原 0.44)
                stage = "up"
            if rel_height < 0.39 and stage == "up": # 放下判定線 (原 0.38)
                stage = "down"
                reps += 1

            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # --- UI 繪製 (左計次，右動作) ---
        cv2.rectangle(frame, (0, 0), (sw, 100), (30, 30, 30), -1)
        cv2.putText(frame, f"REPS: {reps} / {TARGET_REPS}", (40, 65), 1, 2.8, (0, 255, 255), 3)
        cv2.putText(frame, "FORWARD CIRCLES", (sw - 550, 65), 1, 2.8, (255, 255, 255), 3)

        # 離開按鈕
        cv2.rectangle(frame, (0, sh-150), (250, sh), (50, 50, 200), -1)
        cv2.putText(frame, "TOUCH TO", (30, sh-90), 1, 2, (255, 255, 255), 2)
        cv2.putText(frame, "EXIT", (30, sh-40), 1, 2, (255, 255, 255), 2)

        # 示範小視窗
        thumb_w, thumb_h = 400, 300
        thumb = crop_to_fill_top(demo_frames[demo_idx], thumb_w, thumb_h)
        frame[sh-thumb_h-20:sh-20, sw-thumb_w-20:sw-20] = thumb
        cv2.rectangle(frame, (sw-thumb_w-20, sh-thumb_h-20), (sw-20, sh-20), color, 5)

        cv2.imshow(win_name, frame)
        demo_idx = (demo_idx + 1) % len(demo_frames)
        
        if cv2.waitKey(1) & 0xFF == 27 or reps >= TARGET_REPS: break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_trainer()