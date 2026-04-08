import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
#вырезает roi с центральной жилой
#можно настроить параметры поиска центра
def extract_centered_circle(gray_blur):
    maxRadius = 425
    canvas_size = maxRadius*2
    circles = cv2.HoughCircles(
        gray_blur,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=150,
        param1=40,
        param2=20,
        minRadius=420,
        maxRadius=maxRadius,
    )

    if circles is None:
        return None

    circles = np.round(circles[0, :]).astype(int)
    h, w = gray_blur.shape[:2]
    cx, cy, r = max(circles, key=lambda c: c[2])

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), r, 255, -1)

    circle_only = np.zeros_like(gray_blur)
    circle_only[mask > 0] = gray_blur[mask > 0]

    x1 = max(cx - r, 0)
    y1 = max(cy - r, 0)
    x2 = min(cx + r, w)
    y2 = min(cy + r, h)

    roi_cut = circle_only[y1:y2, x1:x2]

    canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)

    rh, rw = roi_cut.shape[:2]

    if rh > canvas_size or rw > canvas_size:
        scale = min(canvas_size / rw, canvas_size / rh)
        new_w = int(rw * scale)
        new_h = int(rh * scale)
        roi_cut = cv2.resize(roi_cut, (new_w, new_h), interpolation=cv2.INTER_AREA)
        rh, rw = roi_cut.shape[:2]

    x_offset = (canvas_size - rw) // 2
    y_offset = (canvas_size - rh) // 2
    canvas[y_offset:y_offset + rh, x_offset:x_offset + rw] = roi_cut

    return canvas, cx, cy

#Процессит изображение в текстуру
def preprocess_roi_for_mog(roi):
    gx = cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3)
    angle = cv2.phase(gx, gy, angleInDegrees=False)
    feat = (angle / (2 * np.pi) * 255).astype(np.uint8)
    return feat

#Постит маску на конечный канвас
def paste_mask_on_black_canvas(mask, cx, cy, canvas):
    """
    mask: numpy array, значения 0/1
    cx, cy: координаты центра, куда надо поставить центр mask на выходном canvas
    out_size: размер выходного canvas (по умолчанию 1024)

    return: numpy array out_size x out_size, черный canvas с вставленной mask
    """
    h, w = mask.shape[:2]

    x0 = int(round(cx - w / 2))
    y0 = int(round(cy - h / 2))
    x1 = x0 + w
    y1 = y0 + h

    dst_x0 = max(0, x0)
    dst_y0 = max(0, y0)
    dst_x1 = min(1024, x1)
    dst_y1 = min(1024, y1)

    if dst_x0 >= dst_x1 or dst_y0 >= dst_y1:
        return canvas

    src_x0 = dst_x0 - x0
    src_y0 = dst_y0 - y0
    src_x1 = src_x0 + (dst_x1 - dst_x0)
    src_y1 = src_y0 + (dst_y1 - dst_y0)

    canvas[dst_y0:dst_y1, dst_x0:dst_x1] = mask[src_y0:src_y1, src_x0:src_x1]
    return canvas

#дебажная. Просто выводит изначальный кадр и результат на экран
def show_result(frame, canvas):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(frame, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(canvas, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

train_paths = sorted(glob.glob("data/train/good/*.png"))
mog = cv2.createBackgroundSubtractorMOG2(
    history=224,
    varThreshold=16,
    detectShadows=False,
)

for epoch in range(1):
    print(f"train epoch {epoch + 1}/5")
    for path in train_paths:
        frame = cv2.imread(path)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        gray = cv2.medianBlur(gray, 5)
        roi, cx, cy = extract_centered_circle(gray)
        roi = preprocess_roi_for_mog(roi)
        
        mog.apply(roi, learningRate = 0.01)


# =========================
# Public API: same idea as model.py
# =========================
def predict(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    gray = cv2.medianBlur(gray, 5)
    roi, cx, cy = extract_centered_circle(gray)
    roi = preprocess_roi_for_mog(roi)
    roi = mog.apply(roi, learningRate = 0)
    canvas = np.zeros((1024, 1024), dtype=roi.dtype)
    canvas = paste_mask_on_black_canvas(roi, cx, cy, canvas)
    show_result(frame, canvas)
    return canvas


# Эта хуйня вырезае круг внешней обмотки отдельно его центрирует и загоняет в mog\
# Потом как получит с нее результат он по центру вклиевает результат на канвас
# Пытаюсь пофиксить то что провода кидает из стороны в сторону. 
# Пока залупа выходит. бо даже так оно видит обычную текстуту обмотки как дефект