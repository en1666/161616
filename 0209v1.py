import cv2
import mediapipe as mp
import numpy as np
import screeninfo
import os

# --- 設定區 ---
VIDEO_FILE = r'C:\Users\User\Desktop\media\0209v\1.手軸彎曲0209.mp4' 
TARGET_REPS_PER_SIDE = 5

# 全域變數：處理離開事件
should_exit = False

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8)

def click_event(event, x, y, flags, param):
    """捕捉滑鼠/觸控點擊事件"""
    global should_exit
    sw, sh = param
    if event == cv2.EVENT_LBUTTONDOWN:
        # 點擊區域：左下角 (寬250, 高150)
        if 0 < x < 250 and (sh - 150) < y < sh:
            print("使用者點擊離開...")
            should_exit = True

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

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

def crop_to_fill(image, target_w, target_h):
    h, w = image.shape[:2]
    if (w/h) > (target_w/target_h):
        new_w = int(h * (target_w/target_h))
        offset = (w - new_w) // 2
        crop = image[:, offset:offset+new_w]
    else:
        new_h = int(w * (target_h/target_w))
        offset = (h - new_h) // 2
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
        print(f"錯誤：找不到影片檔案 {VIDEO_FILE}")
        return

    win_name = 'AI Trainer'
    cv2.namedWindow(win_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    # 預先綁定點擊事件，讓影片播放時也能點
    cv2.setMouseCallback(win_name, click_event, param=(sw, sh))

    # --- 1. 預處理並【全螢幕播放示範】(加入離開按鈕) ---
    demo_cap = cv2.VideoCapture(VIDEO_FILE)
    demo_data, demo_frames = [], []
    
    print("播放示範影片中...")
    while demo_cap.isOpened():
        ret, frame = demo_cap.read()
        if not ret or should_exit: break
        
        display_frame = letterbox_image(frame, sw, sh)
        
        # 在示範影片中也畫上「離開按鈕」
        cv2.rectangle(display_frame, (0, sh-150), (250, sh), (50, 50, 200), -1)
        cv2.putText(display_frame, "TOUCH TO", (30, sh-90), 1, 2, (255, 255, 255), 2)
        cv2.putText(display_frame, "EXIT", (30, sh-40), 1, 2, (255, 255, 255), 2)
        cv2.putText(display_frame, "DEMO MODE", (sw//2-100, 50), 1, 2, (0, 255, 255), 2)

        cv2.imshow(win_name, display_frame)
        if cv2.waitKey(25) & 0xFF == 27: 
            should_exit = True
            break
        
        # 背景處理數據
        res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            p1 = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
            p2 = [lm[mp_pose.PoseLandmark.RIGHT_ELBOW].x, lm[mp_pose.PoseLandmark.RIGHT_ELBOW].y]
            p3 = [lm[mp_pose.PoseLandmark.RIGHT_WRIST].x, lm[mp_pose.PoseLandmark.RIGHT_WRIST].y]
            demo_data.append(calculate_angle(p1, p2, p3))
            demo_frames.append(frame)
    demo_cap.release()

    if should_exit:
        cv2.destroyAllWindows()
        return

    # --- 2. 實時偵測階段 (代碼同前) ---
    cap = cv2.VideoCapture(0)
    current_target = "LEFT" 
    counters = {"LEFT": 0, "RIGHT": 0}
    stage = "wait" 
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
            if current_target == "LEFT":
                s, e, w_pt = mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST
            else:
                s, e, w_pt = mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST

            if lm[e].visibility > 0.7:
                p1, p2, p3 = [lm[s].x, lm[s].y], [lm[e].x, lm[e].y], [lm[w_pt].x, lm[w_pt].y]
                user_angle = calculate_angle(p1, p2, p3)
                is_correct = abs(user_angle - demo_data[demo_idx]) < 25
                color = (0, 255, 0) if is_correct else (0, 0, 255)

                if user_angle > 160: stage = "stretch"
                if user_angle < 45 and stage == "stretch":
                    stage = "flex"
                    counters[current_target] += 1
                    if counters["LEFT"] >= TARGET_REPS_PER_SIDE and current_target == "LEFT":
                        current_target = "RIGHT"
                        stage = "wait"

        # UI 繪製 (狀態、離開按鈕、示範小視窗)
        cv2.rectangle(frame, (0, 0), (sw, 130), (30, 30, 30), -1)
        cv2.putText(frame, f"RIGHT: {counters['LEFT']} / {TARGET_REPS_PER_SIDE}", (40, 50), 1, 2.5, (0, 255, 255), 3)
        cv2.putText(frame, f"LEFT : {counters['RIGHT']} / {TARGET_REPS_PER_SIDE}", (40, 100), 1, 2.5, (0, 255, 255), 3)

        cv2.rectangle(frame, (0, sh-150), (250, sh), (50, 50, 200), -1)
        cv2.putText(frame, "TOUCH TO", (30, sh-90), 1, 2, (255, 255, 255), 2)
        cv2.putText(frame, "EXIT", (30, sh-40), 1, 2, (255, 255, 255), 2)

        thumb_w, thumb_h = 400, 300
        thumb = crop_to_fill(demo_frames[demo_idx], thumb_w, thumb_h)
        frame[sh-thumb_h-20:sh-20, sw-thumb_w-20:sw-20] = thumb
        cv2.rectangle(frame, (sw-thumb_w-20, sh-thumb_h-20), (sw-20, sh-20), color, 5)

        cv2.imshow(win_name, frame)
        demo_idx = (demo_idx + 1) % len(demo_frames)
        if cv2.waitKey(1) & 0xFF == 27: break
        if counters["RIGHT"] >= TARGET_REPS_PER_SIDE: break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_trainer()