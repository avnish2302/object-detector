from fastapi.responses import StreamingResponse
import io
from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

app = FastAPI()


@app.get("/")
def home():
    return {"message": "Cleanliness detector API running"}


# -------- IMAGE ALIGNMENT (ORB) --------
def align_images(img1, img2):

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(5000)

    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # keep best 15% matches
    matches = matches[: int(len(matches) * 0.15)]

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC)

    height, width = img1.shape[:2]

    aligned = cv2.warpPerspective(img2, H, (width, height))

    return aligned


# -------- MAIN API --------
@app.post("/compare")
async def compare_images(
    baseline: UploadFile = File(...),
    current: UploadFile = File(...)
):

    baseline_bytes = await baseline.read()
    current_bytes = await current.read()

    baseline_img = cv2.imdecode(
        np.frombuffer(baseline_bytes, np.uint8),
        cv2.IMREAD_COLOR
    )

    current_img = cv2.imdecode(
        np.frombuffer(current_bytes, np.uint8),
        cv2.IMREAD_COLOR
    )

    # -------- ALIGN IMAGES --------
    current_img = align_images(baseline_img, current_img)

    # -------- DOWNSCALE (reduces noise) --------
    scale = 0.5
    baseline_img = cv2.resize(baseline_img, None, fx=scale, fy=scale)
    current_img = cv2.resize(current_img, None, fx=scale, fy=scale)

    # -------- GRAYSCALE --------
    baseline_gray = cv2.cvtColor(baseline_img, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)

    # -------- NOISE REDUCTION --------
    baseline_gray = cv2.GaussianBlur(baseline_gray, (15, 15), 0)
    current_gray = cv2.GaussianBlur(current_gray, (15, 15), 0)

    # -------- SSIM DIFFERENCE --------
    score, ssim_diff = ssim(baseline_gray, current_gray, full=True)
    ssim_diff = (ssim_diff * 255).astype("uint8")

    # -------- ABSOLUTE DIFFERENCE --------
    abs_diff = cv2.absdiff(baseline_gray, current_gray)
    abs_diff = cv2.GaussianBlur(abs_diff, (7,7), 0)

    # -------- THRESHOLD BOTH --------
    _, ssim_thresh = cv2.threshold(
        ssim_diff,
        200,
        255,
        cv2.THRESH_BINARY_INV
    )

    _, abs_thresh = cv2.threshold(
        abs_diff,
        25,
        255,
        cv2.THRESH_BINARY
    )

    # -------- COMBINE MASKS --------
    combined = cv2.bitwise_and(ssim_thresh, abs_thresh)

    # -------- MORPHOLOGICAL CLEANUP --------
    kernel = np.ones((9,9), np.uint8)

    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    combined = cv2.dilate(combined, kernel, iterations=2)

    # -------- FIND CONTOURS --------
    contours, _ = cv2.findContours(
        combined,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    result = current_img.copy()

    h, w = result.shape[:2]

    # -------- FILTER CONTOURS --------
    for c in contours:

        area = cv2.contourArea(c)

        if area < 1000:
            continue

        x, y, w_box, h_box = cv2.boundingRect(c)

        # ignore tiny boxes
        if w_box < 20 or h_box < 20:
            continue

        aspect_ratio = w_box / float(h_box)

        # remove long thin shapes
        if aspect_ratio > 6 or aspect_ratio < 0.15:
            continue

        # solidity filter
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)

        if hull_area == 0:
            continue

        solidity = float(area) / hull_area

        if solidity < 0.4:
            continue

        # ignore borders (alignment artifacts)
        if x < 20 or y < 20 or x + w_box > w - 20 or y + h_box > h - 20:
            continue

        # draw detection box
        cv2.rectangle(
            result,
            (x, y),
            (x + w_box, y + h_box),
            (0, 0, 255),
            3
        )

    # -------- RETURN IMAGE --------
    _, buffer = cv2.imencode(".jpg", result)

    return StreamingResponse(
        io.BytesIO(buffer.tobytes()),
        media_type="image/jpeg"
    )