# Age and Gender Detection

This project demonstrates real-time age and gender detection using OpenCV and pre-trained deep learning models. The script captures video input from a webcam, detects faces, and predicts the age range and gender of the detected faces.

## Features
- Detects faces in a video feed using OpenCV's Haar Cascade.
- Predicts age range and gender for each detected face using pre-trained deep learning models.
- Displays predictions and bounding boxes in real-time.

## Requirements
- Python 3.6+
- OpenCV

## Installation

1. Install required Python packages:
   ```bash
   pip install opencv-python opencv-python-headless numpy
   ```

2. Download the pre-trained models and network structure files:
   - **Age Model**: [age_net.caffemodel](https://github.com/spmallick/learnopencv/tree/master/AgeGender/models/age_net.caffemodel)
   - **Age Deploy Prototxt**: [age_deploy.prototxt](https://github.com/spmallick/learnopencv/tree/master/AgeGender/models/age_deploy.prototxt)
   - **Gender Model**: [gender_net.caffemodel](https://github.com/spmallick/learnopencv/tree/master/AgeGender/models/gender_net.caffemodel)
   - **Gender Deploy Prototxt**: [gender_deploy.prototxt](https://github.com/spmallick/learnopencv/tree/master/AgeGender/models/gender_deploy.prototxt)

3. Ensure the Haar Cascade XML file is available for face detection. It is typically included with OpenCV:
   ```
   cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
   ```

## Running the Project

1. Clone this repository or save the provided code as `age_gender_detection.py`.

2. Place the downloaded models and prototxt files in the same directory as the script or update the paths in the script accordingly.

3. Run the script:
   ```bash
   python age_gender_detection.py
   ```

4. The webcam window will open, showing real-time predictions of age and gender for detected faces. Press `q` to quit the application.

## Code Explanation

1. **Face Detection**:
   - Uses Haar Cascade to detect faces in the video feed.

2. **Age and Gender Prediction**:
   - Extracts the detected face and processes it using the DNN module in OpenCV.
   - Uses pre-trained Caffe models to predict age and gender.

3. **Results Display**:
   - Draws bounding boxes around faces and overlays predicted age and gender labels.

## Example Output
- Detected face with bounding box.
- Predicted labels, e.g., `Male, (25-32)`.

## Customization
- Update the webcam source by changing `cv2.VideoCapture(0)` to a different camera index or video file path.
- Modify the bounding box color and font style in the `cv2.rectangle` and `cv2.putText` functions.

## References
- [LearnOpenCV Age and Gender Detection](https://www.learnopencv.com/age-gender-classification-using-opencv-deep-learning-c-python/)
- OpenCV Documentation: [DNN Module](https://docs.opencv.org/master/d6/d0f/group__dnn.html)

## License
This project is for educational purposes and uses pre-trained models publicly available under their respective licenses.
