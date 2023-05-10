# app.py
from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
from werkzeug.utils import secure_filename


import cv2
import glob
import numpy as np
from dom import DOM
import imutils
import easyocr
from matplotlib import pyplot as plt

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    files = request.files.getlist('file')
    print('files ::', files)
    file_names = []
    for file in files:
        print('loop file', file)
        # if file.filename == '':
        #     flash('Please select images first')
        #     return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_names.append(filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print('upload_image filename: ' + filename)
            flash('selected vehicle uploaded successfully')
        else:
            flash('Allowed image types are - png, jpg, jpeg')
            return redirect(request.url)
    return render_template('index.html', filename=filename)


@app.route('/display/<filename>')
def display_image(filename):
    # print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


@app.route('/cards')
def display_cards():
    return render_template('cards.html')


@app.route('/')
def detector():
    class VehicleDetector:

        def __init__(self):
            # Load Network
            net = cv2.dnn.readNet("dnn_model/yolov4.weights",
                                "dnn_model/yolov4.cfg")
            self.model = cv2.dnn_DetectionModel(net)
            self.model.setInputParams(size=(832, 832), scale=1 / 255)

            # Allow classes containing Vehicles only
            self.classes_allowed = [2, 3, 5, 6, 7]

        def detect_vehicles(self, img):
            # Detect Objects
            vehicles_boxes = []
            class_ids, scores, boxes = self.model.detect(img, nmsThreshold=0.4)
            for class_id, score, box in zip(class_ids, scores, boxes):
                if score < 0.5:
                    # Skip detection with low confidence
                    continue

                if class_id in self.classes_allowed:
                    vehicles_boxes.append(box)

            return vehicles_boxes


    # Initialize DOM
    iqa = DOM()

    # Set the threshold
    threshold = 0.94

    # Initialize the vehicle detector
    vd = VehicleDetector()

    # Initialize EasyOCR reader
    reader = easyocr.Reader(['en'])

    # Load images from a folder
    images_folder = glob.glob("static/uploads/Capture1.jpeg")

    # Loop through all the images
    for img_path in images_folder:
        img = cv2.imread(img_path)

        # Image quality check
        while True:
            score = iqa.get_sharpness(img)
            if score < threshold:
                print(f"Image quality too low for image {img_path}, reupload.")
                break
                # Reload image (in real use case, ask the user to re-upload)
                # img = cv2.imread(img_path)
            else:
                break

        # Vehicle detection check
        while True:
            vehicle_boxes = vd.detect_vehicles(img)
            if not vehicle_boxes:
                print(f"No vehicle found in image {img_path}, reupload.")
                break
                # Reload image (in real use case, ask the user to re-upload)
            #     img = cv2.imread(img_path)
            else:
                break

        # Number plate recognition
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bfilter = cv2.bilateralFilter(gray, 11, 17, 17)  # Noise reduction
        edged = cv2.Canny(bfilter, 30, 200)  # Edge detection

        keypoints = cv2.findContours(
            edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(keypoints)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        location = None
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 10, True)
            if len(approx) == 4:
                location = approx
                break

        if location is not None:
            mask = np.zeros(gray.shape, np.uint8)

            try:
                new_image = cv2.drawContours(mask, [location], 0, 255, -1)
                print("contour found")
            except cv2.error as e:
                print("Error drawing contours:", e)
                new_image = mask.copy()

            new_image = cv2.bitwise_and(img, img, mask=mask)

            # plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))

            (x, y) = np.where(mask == 255)
            (x1, y1) = (np.min(x), np.min(y))
            (x2, y2) = (np.max(x), np.max(y))
            cropped_image = gray[x1:x2 + 1, y1:y2 + 1]

            # cropped num plate
            plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

            reader = easyocr.Reader(['en'])
            result = reader.readtext(cropped_image)
            print("here", result)

            try:
                text = result[0][-2]
                print(text)
                font = cv2.FONT_HERSHEY_SIMPLEX
                res = cv2.putText(img, text=text, org=(
                    approx[0][0][0], approx[1][0][1] + 60), fontFace=font, fontScale=1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
                res = cv2.rectangle(img, tuple(approx[0][0]), tuple(
                    approx[2][0]), (0, 255, 0), 3)
            except IndexError as e:
                print("Error extracting text from result:", e)
                res = img.copy()

        else:
            print("Number plate not found in image", img_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081, debug=True)
