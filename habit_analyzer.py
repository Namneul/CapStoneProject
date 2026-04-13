import cv2
import mediapipe as mp
import time
import math

warning_count = 0
warning_log = []
start_time = time.time()


FINGERS = {
    4:  (3, 2),    # 엄지: tip=4,  IP=3,  MCP=2
    8:  (6, 5),    # 검지: tip=8,  PIP=6, MCP=5
    12: (10, 9),   # 중지: tip=12, PIP=10, MCP=9
    16: (14, 13),  # 약지: tip=16, PIP=14, MCP=13
    20: (18, 17),  # 소지: tip=20, PIP=18, MCP=17
}

FACE_REGIONS = {
    "forehead": [10, 67, 109, 338, 297, 69, 104, 108, 299, 333],
    "head_top": [10, 67, 109, 338, 297, 69, 104, 108, 299, 333],
    "left_eye":    [33, 133, 159, 145, 160, 144],
    "right_eye":   [362, 263, 386, 374, 387, 373],
    "nose":        [1, 2, 3, 4, 5, 6, 195, 197, 98, 327, 48, 278],
    "mouth":       [13, 14, 78, 308, 61, 291],
    "chin":        [152, 377, 148, 176, 400, 175, 199],
    "left_cheek":  [234, 93, 132, 123, 116, 117, 50, 205, 187, 147, 213],
    "right_cheek": [454, 323, 361, 352, 345, 346, 280, 425, 411, 376, 433],
}


REGION_MARGINS = {
    "forehead": (0.02, 0.02, 0.02, 0.02),
    "head_top":    (0.05, 0.15, 0.05, 0.00),
    "left_eye":    (0.015, 0.015, 0.015, 0.015),
    "right_eye":   (0.015, 0.015, 0.015, 0.015),
    "nose":        (0.03, 0.03, 0.03, 0.03),
    "mouth":       (0.02, 0.02, 0.02, 0.02),
    "chin":        (0.03, 0.03, 0.03, 0.03),
    "left_cheek":  (0.04, 0.04, 0.04, 0.04),
    "right_cheek": (0.04, 0.04, 0.04, 0.04),
}

prev_touched = None #이전에 만지고 있던 얼굴 부위(엣지 트리거)
region_counts = {r: 0 for r in FACE_REGIONS}
region_log = []  # (elapsed, side, region)


LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
EAR_THRESHOLD = 0.22 #깜빡임 경계값
blink_count = 0
blink_log = []
eye_closed = False

# MediaPipe 초기화
mp_face = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

face = mp_face.FaceMesh(
    max_num_faces = 1,
    refine_landmarks = True,
    min_detection_confidence = 0.7,
    min_tracking_confidence = 0.5
)

hands = mp_hands.Hands(
    max_num_hands = 2,
    min_detection_confidence = 0.7,
    min_tracking_confidence = 0.5
)

pose = mp_pose.Pose(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# 몸 흔들거림 — 양쪽 어깨 중심점 x좌표를 매 프레임 기록
# 최근 1초(30프레임)간의 이동 범위가 threshold를 넘으면 "흔들림"
shoulder_history = []
SWAY_WINDOW = 30          # 30fps 기준 1초
SWAY_THRESHOLD = 0.015    # 정규화 좌표 기준
sway_count = 0
sway_log = []
was_swaying = False

# 시선 불안정 — iris 랜드마크로 시선 방향 판정
# 눈 안에서 홍채가 좌/우/중앙 어디에 있는지 비율로 계산
gaze_shift_count = 0
gaze_shift_log = []
last_gaze = "center"


# ================================================================
#  유틸 함수
# ================================================================


#깜빡임 함수
def calc_ear(landmarks, eye_indices):
    def dist(a, b):
        return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)
    p = [landmarks.landmark[i] for i in eye_indices]
    vertical1 = dist(p[1], p[5])
    vertical2 = dist(p[2], p[4])
    horizontal = dist(p[0], p[3])

    if horizontal == 0:
        return 0.3
    return (vertical1 + vertical2) / (2.0 * horizontal)

#손가락 폈는지 쥐었는지 판별
def is_finger_extended(hand_landmarks, tip_idx):
    pip_idx, mcp_idx = FINGERS[tip_idx]

    tip = hand_landmarks.landmark[tip_idx]
    pip = hand_landmarks.landmark[pip_idx]
    mcp = hand_landmarks.landmark[mcp_idx]

    def dist(a, b):
        return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)

    return dist(tip, mcp) > dist(pip, mcp)

def get_region_bbox(face_landmarks, indices, region_name):
    ml, mt, mr, mb = REGION_MARGINS[region_name]
    xs = [face_landmarks.landmark[i].x for i in indices]
    ys = [face_landmarks.landmark[i].y for i in indices]
    return (
        max(0, min(xs) - ml),
        max(0, min(ys) - mt),
        min(1, max(xs) + mr),
        min(1, max(ys) + mb),
    )


# ================================================================
#  웹캠 시작
# ================================================================

#OpenCV 웹캠 설정
cap = cv2.VideoCapture(0) #기본 내장카메라(0)로 카메라 설정
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while cap.isOpened():
    ret, frame = cap.read() #매 프레임 하나씩 가져옴
    if not ret:
        break

    frame = cv2.flip(frame, 1) # 거울처럼 화면 반전
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #MediaPipe는 RGB값을 받는데 OpenCV는 BGR로 읽어서 변환.
    face_res = face.process(rgb) #FaceMesh 모델로 468개 랜드마크 추출
    hand_res = hands.process(rgb)
    pose_res = pose.process(rgb)

    elapsed = time.time() - start_time
    now = time.time()

    # ── 몸 흔들거림 감지 ──
    if pose_res.pose_landmarks:
        pl = pose_res.pose_landmarks
        # 양쪽 어깨 중심의 x좌표 추적
        ls = pl.landmark[11]   # 왼쪽 어깨
        rs = pl.landmark[12]   # 오른쪽 어깨
        center_x = (ls.x + rs.x) / 2

        shoulder_history.append(center_x)
        if len(shoulder_history) > SWAY_WINDOW:
            shoulder_history.pop(0)

        # 최근 1초간 어깨 중심의 좌우 이동 범위
        if len(shoulder_history) == SWAY_WINDOW:
            sway_range = max(shoulder_history) - min(shoulder_history)
            currently_swaying = sway_range > SWAY_THRESHOLD

            # 엣지 트리거: 안 흔들다가 흔들기 시작할 때만 카운트
            if currently_swaying and not was_swaying:
                sway_count += 1
                sway_log.append(elapsed)
                print(f"  [SWAY] #{sway_count} ({int(elapsed)}s)")

            was_swaying = currently_swaying

            if currently_swaying:
                cv2.putText(frame, "SWAYING", (20, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)

    # ── 얼굴 처리: 깜빡임 + 시선 ──
    if face_res.multi_face_landmarks:
        fl = face_res.multi_face_landmarks[0]

        # 얼굴 영역 bbox 계산
        xs = [lm.x for lm in fl.landmark]
        ys = [lm.y for lm in fl.landmark]
        margin = 0.04
        x_min = max(0, min(xs) - margin)
        x_max = min(1, max(xs) + margin)
        y_min = max(0, min(ys) - margin)
        y_max = min(1, max(ys) + margin)

        # ── 눈 깜빡임 ──
        left_ear = calc_ear(fl, LEFT_EYE)
        right_ear = calc_ear(fl, RIGHT_EYE)
        avg_ear = (left_ear + right_ear) / 2

        if avg_ear < EAR_THRESHOLD:
            if not eye_closed:
                eye_closed = True
        else:
            if eye_closed:
                blink_count += 1
                blink_log.append(elapsed)
                eye_closed = False

        # ── 시선 불안정 ──
        # 양쪽 눈 중심 계산
        e1_cx = (fl.landmark[33].x + fl.landmark[133].x) / 2
        e2_cx = (fl.landmark[362].x + fl.landmark[263].x) / 2
        # iris를 가까운 눈에 매칭 (flip 후 배정이 뒤바뀔 수 있어서)
        i1 = fl.landmark[468]
        i2 = fl.landmark[473]

        if abs(i1.x - e1_cx) < abs(i1.x - e2_cx):
            pairs = [(i1, 33, 133), (i2, 362, 263)]
        else:
            pairs = [(i1, 362, 263), (i2, 33, 133)]

        # 각 눈에서 iris 위치 비율 계산 (0=왼쪽, 0.5=정면, 1=오른쪽)
        ratios = []
        for iris, c1, c2 in pairs:
            left_x = min(fl.landmark[c1].x, fl.landmark[c2].x)
            right_x = max(fl.landmark[c1].x, fl.landmark[c2].x)
            eye_w = right_x - left_x
            if eye_w > 1e-6:
                ratios.append((iris.x - left_x) / eye_w)

        avg_ratio = sum(ratios) / len(ratios) if ratios else 0.5

        if avg_ratio < 0.45:
            current_gaze = "left"
        elif avg_ratio > 0.55:
            current_gaze = "right"
        else:
            current_gaze = "center"

        # 디버깅용 — 확인 후 삭제
        cv2.putText(frame, f"Gaze: {avg_ratio:.3f} | {current_gaze}",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if current_gaze != last_gaze and current_gaze != "center":
            gaze_shift_count += 1
            gaze_shift_log.append((elapsed, current_gaze))
            print(f"  [GAZE] {current_gaze} ({int(elapsed)}s)")
        last_gaze = current_gaze


    # ── 터치 감지 ──
    touched_region = None
    detected_side = ""

    if face_res.multi_face_landmarks and hand_res.multi_hand_landmarks:
        fl = face_res.multi_face_landmarks[0]

        for idx, hl in enumerate(hand_res.multi_hand_landmarks):
            if hand_res.multi_handedness:
                side = hand_res.multi_handedness[idx].classification[0].label
            else:
                side = "Unknown"

            # 1차: 펴진 손가락 끝으로만 판정 (정밀)
            fingertips = [ci for ci in [4, 8, 12, 16, 20]
                          if is_finger_extended(hl, ci)]

            # 매칭되는 모든 부위 수집 (bbox 면적과 함께)
            # 가장 작은 bbox = 가장 구체적인 부위 선택
            hits = []
            for ci in fingertips:
                lm = hl.landmark[ci]
                for region_name, indices in FACE_REGIONS.items():
                    rx_min, ry_min, rx_max, ry_max = get_region_bbox(fl, indices, region_name)
                    if (rx_min <= lm.x <= rx_max and
                            ry_min <= lm.y <= ry_max):
                        area = (rx_max - rx_min) * (ry_max - ry_min)
                        hits.append((area, region_name))

            # 2차: 손가락으로 안 잡혔으면 손바닥으로 판정 (턱 괴기 등)
            if not hits:
                for ci in [0, 5, 9]:
                    lm = hl.landmark[ci]
                    for region_name, indices in FACE_REGIONS.items():
                        rx_min, ry_min, rx_max, ry_max = get_region_bbox(fl, indices, region_name)
                        if (rx_min <= lm.x <= rx_max and
                                ry_min <= lm.y <= ry_max):
                            area = (rx_max - rx_min) * (ry_max - ry_min)
                            hits.append((area, region_name))

            # 가장 작은 bbox = 가장 구체적인 부위
            if hits:
                hits.sort(key=lambda x: x[0])
                touched_region = hits[0][1]
                detected_side = side

            # 카운트 처리 (엣지 트리거)
            if touched_region:
                if touched_region != prev_touched:
                    region_counts[touched_region] += 1
                    region_log.append((elapsed, side, touched_region))
                    print(f"  [TOUCH] {touched_region} - {side} hand ({int(elapsed)}s)")
                    warning_count += 1
                break

    # 경고 표시
    if touched_region:
        cv2.rectangle(frame, (2, 2), (w - 3, h - 3), (0, 0, 255), 4)
        cv2.putText(frame, f"TOUCHING: {touched_region.upper()}",
                    (w // 2 - 200, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    prev_touched = touched_region
    cv2.imshow("Habit Analyzer", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# ================================================================
#  종료 요약
# ================================================================

total_sec = time.time() - start_time
cap.release()

print("\n" + "=" * 50)
print("  SESSION SUMMARY")
print("=" * 50)
mins, secs = divmod(int(total_sec), 60)
print(f"  Duration      : {mins:02d}:{secs:02d}")

# 부위별 터치
print(f"\n  [Face Touch] Total: {warning_count}")
for region, cnt in region_counts.items():
    if cnt > 0:
        print(f"    {region:>12s}: {cnt} times")
for i, (t, side, region) in enumerate(region_log):
    m, s = divmod(int(t), 60)
    print(f"    #{i+1:02d}  {m:02d}:{s:02d}  {region} ({side} hand)")

# 깜빡임
bpm = blink_count / (total_sec / 60) if total_sec > 0 else 0
print(f"\n  [Blink] Total: {blink_count} ({bpm:.1f} /min)")
if bpm > 35:
    print(f"    ⚠ Above normal (15-35/min) — possible nervousness")

# 시선 불안정
gaze_per_min = gaze_shift_count / (total_sec / 60) if total_sec > 0 else 0
print(f"\n  [Gaze] Shifts: {gaze_shift_count} ({gaze_per_min:.1f} /min)")
if gaze_per_min > 20:
    print(f"    ⚠ Unstable gaze — may indicate nervousness")
for i, (t, direction) in enumerate(gaze_shift_log):
    m, s = divmod(int(t), 60)
    print(f"    #{i+1:02d}  {m:02d}:{s:02d}  {direction}")

# 몸 흔들거림
print(f"\n  [Sway] Episodes: {sway_count}")
for i, t in enumerate(sway_log):
    m, s = divmod(int(t), 60)
    print(f"    #{i+1:02d}  {m:02d}:{s:02d}")

print("=" * 50)

cv2.destroyAllWindows()
