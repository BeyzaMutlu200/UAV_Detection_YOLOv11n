import cv2
from ultralytics import YOLO
import cvzone

# Modeli yükle
model = YOLO("C:\\Users\\Beyza\\source\\uav_model\\runs1\\detect\\train10\\weights\\best.pt")

# Video dosyasını aç
video_path = "C:\\Users\\Beyza\\source\\uav_model\\uav_new.mp4"
cap = cv2.VideoCapture(video_path)

cv2.namedWindow("YOLO Video Kilitlenme Takibi", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLO Video Kilitlenme Takibi", 1280, 720)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Video bitti veya okunamadı.")
        break

    frame = cv2.resize(frame, (1280, 720))
    results = model(frame)

    frame_height, frame_width = frame.shape[:2]
    target_point = (frame_width // 2, frame_height // 2)

    # Yeşil kenarlık (tüm ekranın çevresi)
    cv2.rectangle(frame, (0, 0), (frame_width - 1, frame_height - 1), (0, 255, 0), 1)

    # Sarı hedef kutusu (AV alanı)
    margin_y = int(frame_height * 0.10)
    margin_x = int(frame_width * 0.25)
    av_top_left = (margin_x, margin_y)
    av_bottom_right = (frame_width - margin_x, frame_height - margin_y)

    frame = cvzone.cornerRect(
        frame,
        (av_top_left[0], av_top_left[1], av_bottom_right[0] - av_top_left[0], av_bottom_right[1] - av_top_left[1]),
        l=30,
        t=5,
        rt=1,
        colorR=(0, 255, 255),
        colorC=(0, 255, 255)
    )

    kilitlenme_basarili = False

    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                cv2.line(frame, target_point, (cx, cy), (0, 0, 255), 1)
                cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

                bbox_width = x2 - x1
                bbox_height = y2 - y1
                margin_bbox_x = int(bbox_width * 0.05)
                margin_bbox_y = int(bbox_height * 0.05)

                ah_top_left = (x1 - margin_bbox_x, y1 - margin_bbox_y)
                ah_bottom_right = (x2 + margin_bbox_x, y2 + margin_bbox_y)

                cv2.rectangle(frame, ah_top_left, ah_bottom_right, (0, 0, 255), 1)

                # Etiket bilgisi ve güven skoru
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = model.names[cls_id]
                label = f"{class_name} {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                in_ah = (ah_top_left[0] >= av_top_left[0] and ah_top_left[1] >= av_top_left[1] and
                         ah_bottom_right[0] <= av_bottom_right[0] and ah_bottom_right[1] <= av_bottom_right[1])

                if in_ah:
                    kilitlenme_basarili = True

    cv2.circle(frame, target_point, 3, (0, 0, 255), -1)

    kilit_text = "Kilitlenme: BASARILI" if kilitlenme_basarili else "Kilitlenme: BASARISIZ"
    kilit_color = (0, 255, 0) if kilitlenme_basarili else (0, 0, 255)
    cv2.putText(frame, kilit_text, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, kilit_color, 2)

    cv2.imshow("YOLO Video Kilitlenme Takibi", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
