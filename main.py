from fastapi.responses import (StreamingResponse)                               # used to return an image file directly from the API instead of JSON
import io                                                                       # used to convert image bytes into a stream object for FASTAPI response
from fastapi import (FastAPI, UploadFile, File)                                 # FastAPI : create API server, UploadFile : handle file upload, File : specify uploaded files
import cv2                                                                      # OpenCV library used for computer vision operations
import numpy as np                                                              # NumPy handles matrix operations and image arrays
from skimage.metrics import (structural_similarity as ssim,)                    # this imports SSIM (Structural Similarity Index). SSIM measure how similar two images are. Value range : 1.0 = identical images, 0 = completely different

app = (FastAPI())                                                               # creates the API server instance. Equivalent to const app = express() in node.js

@app.get("/")                                                                   # Root endpoint : Defines an API endpoint : GET /
def home():
    return {"message": "Cleanliness detector API running"}                      # when someone opens : http://localhost:8000/ they get : {"message" : "Cleanliness detector API running"}



def align_images(img1, img2):                                                   # this function aligns current image with baseline image

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)                              # convert both images to grayscale. Reason : color information not needed for alignment. Grayscale is faster
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(5000)                                                  # ORB feature detector. ORB = Oriented FAST and Rotated BRIEF. It detects keypoints (important visual features)

    kp1, des1 = orb.detectAndCompute(gray1, None)                               # Detect keypoints and descriptors : kp1, des1 = orb.detectAndCompute(gray1, None). Outputs:      kp(variable) : kepoints(meaning). des(variable) descriptors(meaning)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)                  # BFMatcher = Brute Force matcher. Matches features between the two images. Hamming distance is used because ORB descriptors are binary

    matches = matcher.match(des1, des2)                                         # matches keypoints between images. Example : corner in baseline <-> corner in current

    matches = sorted(matches, key=lambda x: x.distance)                         # Sort matches by quality. Lower distance = better match. Sorting keeps best matches first
    matches = matches[: int(len(matches) * 0.15)]                               # Only top 15% matches are used. Reason : remove bad matches, improve alignment accuracy

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)  # Extract keypoint coordinates. Exact coordinates of matched points in baseline image
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)  # coordinates in current image

    H, _ = cv2.findHomography( pts2, pts1, cv2.RANSAC)                          # Homography = transformation matrix. It describes how to transform image2 to align with image1. RANSAC removes incorrect matches.

    height, width = img1.shape[:2]                                              # Extract height and width of baseline image

    aligned = cv2.warpPerspective(img2, H, (width, height))                     # wrap image. Transforms the current image using the homography matrix. Result : current image aligned with baseline

    return aligned                                                              # return aligned image. Now both images have same perspective



@app.post("/compare")                                                           # Defines endpoint : POST /compare (used to upload images)
async def compare_images(
    baseline: UploadFile = File(...),                                           # user uploads baseline image, current image
    current: UploadFile = File(...),
):

    baseline_bytes = (await baseline.read())                                    # read image bytes. Reads uploaded files as binary data.
    current_bytes = await current.read()

    baseline_img = (
        cv2.imdecode(np.frombuffer(baseline_bytes, np.uint8), cv2.IMREAD_COLOR) # convert bytes to image. Steps : bytes -> numpy array -> image
    )

    current_img = cv2.imdecode(np.frombuffer(current_bytes, np.uint8), cv2.IMREAD_COLOR)

    current_img = align_images(baseline_img, current_img)                       # align images. Corrects camera movement

    scale = 0.5
    baseline_img = cv2.resize(baseline_img, None, fx=scale, fy=scale)
    current_img = cv2.resize(current_img, None, fx=scale, fy=scale)

    baseline_gray = cv2.cvtColor(baseline_img, cv2.COLOR_BGR2GRAY)              # convert to grayscale. SSIM works on grayscale images
    current_gray = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)

    baseline_gray = cv2.GaussianBlur(baseline_gray, (15, 15), 0)                  # noise reduction : gaussian blur removes small noise. Helps avoid detecting: tiny texture changes, lighting noise
    current_gray = cv2.GaussianBlur(current_gray, (15, 15), 0)

    score, diff = ssim( baseline_gray, current_gray, full=True)                 # SSIM comparison. Outputs : score(variable) : similarity score(meaning). diff(variable) : difference map(meaning)

    diff = (diff * 255).astype("uint8")                                         # convert diff to image. SSIM output is float between : 0 - 1. Converted to image scale : 0 -255

    edges1 = cv2.Canny(baseline_gray, 60, 150)
    edges2= cv2.Canny(current_gray, 60, 150)

    edges = cv2.bitwise_or(edges1, edges2)
    diff[edges > 0] = 0

    thresh = cv2.threshold(diff, 200, 255, cv2.THRESH_BINARY_INV)[1]            # threshold difference. Stricter threshold (ignore tiny brightness changes). Creates binary image : white =difference, black = no difference. Threshold = 200 mean ignore small brightness changes

    kernel = np.ones((9, 9), np.uint8)                                        # morphological cleanup. Creates structuring element

    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)                   # removes small noise
    thresh = cv2.dilate(thresh, kernel, iterations=2)                           # expands detected areas slightly

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # contours represent objects or changed regions

    result = (current_img.copy())                                               # copy image for drawing. We draw rectangles on this copy

    h, w = result.shape[:2]                                                     # needed for border filtering


    for c in contours:

        area = cv2.contourArea(c)

                                                                                      
        if area < 1000:                                                         # ignore very small objects
            continue

        x, y, w_box, h_box = cv2.boundingRect(c)

        # ignore very small bounding boxes
        if w_box < 20 or h_box < 20:
            continue

        
        aspect_ratio = w_box / float(h_box)                                     # ignore very thin shapes (like notebook lines)
        if aspect_ratio > 6 or aspect_ratio < 0.15:
            continue

        
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

        # draw bounding box
        cv2.rectangle(result, (x, y), (x + w_box, y + h_box), (0, 0, 255), 3)

    # encode image AFTER loop
    _, buffer = cv2.imencode(".jpg", result)

    return StreamingResponse(
        io.BytesIO(buffer.tobytes()),
        media_type="image/jpeg"
    )