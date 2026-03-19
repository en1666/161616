import cv2
import mediapipe as mp
import numpy as np
import screeninfo
import os

# --- 設定區 ---
VIDEO_FILE = r'C:\Users\User\Desktop\media\0209v\2.手指伸展0209.mp4' 
TARGET_REPS_PER_SIDE = 5

# 全域變數：處理離開事件
should_exit = False

# 使用 MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6)
mp_drawing = mp.solutions.drawing_utils

def click_event(event, x, y, flags, param):
    """捕捉滑鼠/觸控點擊事件"""
    global should_exit
    sw, sh = param
    if event == cv2.EVENT_LBUTTONDOWN:
        if 0 < x < 250 and (sh - 150) < y < sh:
            should_exit = True

def get_hand_extension(hand_landmarks):
    """計算手指伸展程度"""
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    tips = [
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP
    ]
    dist_sum = sum([np.sqrt((hand_landmarks.landmark[t].x - wrist.x)**2 + 
                             (hand_landmarks.landmark[t].y - wrist.y)**2) for t in tips])
    return dist_sum / len(tips)

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
        offset = int((h - new_h) * 0.2)
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
        res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if res.multi_hand_landmarks:
            demo_data.append(get_hand_extension(res.multi_hand_landmarks[0]))
            demo_frames.append(frame)
    demo_cap.release()

    if should_exit: return

    # --- 2. 實時計次 (修正鏡像對應關係) ---
    cap = cv2.VideoCapture(0)
    
    # 【關鍵修正】鏡像模式下：
    # 現實左手 = MediaPipe 標籤 "Right"
    # 現實右手 = MediaPipe 標籤 "Left"
    current_target = "Right"  # 優先偵測現實中的左手
    counters = {"Right": 0, "Left": 0} 
    stage = "wait" 
    demo_idx = 0

    while cap.isOpened() and not should_exit:
        success, frame = cap.read()
        if not success: break
        
        frame = cv2.flip(frame, 1) 
        frame = letterbox_image(frame, sw, sh)
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        color = (0, 0, 255)

        if results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                label = results.multi_handedness[i].classification[0].label
                
                # 只有當前目標手才進行計次
                if label == current_target:
                    ext = get_hand_extension(hand_landmarks)
                    is_correct = abs(ext - demo_data[demo_idx]) < 0.15
                    color = (0, 255, 0) if is_correct else (0, 0, 255)

                    if ext < 0.25: stage = "close"
                    if ext > 0.35 and stage == "close":
                        stage = "open"
                        counters[label] += 1 
                        
                        # 當左手(Right標籤)做完，切換到右手(Left標籤)
                        if current_target == "Right" and counters["Right"] >= TARGET_REPS_PER_SIDE:
                            current_target = "Left"
                            stage = "wait"
                            
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # --- UI 顯示修正 ---
        cv2.rectangle(frame, (0, 0), (sw, 130), (30, 30, 30), -1)
        # 顯示為 LEFT，但讀取的是 counters["Right"] (對應現實左手)
        cv2.putText(frame, f"RIGHT : {counters['Right']} / {TARGET_REPS_PER_SIDE}", (40, 50), 1, 2.5, (0, 255, 255), 3)
        # 顯示為 RIGHT，但讀取的是 counters["Left"] (對應現實右手)
        cv2.putText(frame, f"Left: {counters['Left']} / {TARGET_REPS_PER_SIDE}", (40, 100), 1, 2.5, (0, 255, 255), 3)

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
        if cv2.waitKey(1) & 0xFF == 27: break
        # 右手(Left標籤)完成後結束
        if counters["Left"] >= TARGET_REPS_PER_SIDE: break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_trainer()