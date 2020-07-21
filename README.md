# Computer Pointer Controller

Computer Pointer Controller software is used for controlling mouse pointer motion by eye direction and also estimated head pose. This application takes video or camera stream as input and then the app predicts eye direction and head position, and moves the mouse pointers based on that estimate.

## Project Set Up and Installation

#### 1. Source Environment

Run the following command on a new terminal window.

```
cd <openvino directory>\bin\setupvars.bat
```

#### 2. Download the models

Enter the following commands to download each model.

 -  [Face Detection Model](https://docs.openvinotoolkit.org/latest/_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html)


    ```
    python <openvino directory>/deployment_tools/tools/model_downloader/downloader.py --name "face-detection-adas-binary-0001"
    ```

- [Facial Landmarks Detection Model](https://docs.openvinotoolkit.org/latest/_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html)

    ```
    python <openvino directory>/deployment_tools/tools/model_downloader/downloader.py --name "landmarks-regression-retail-0009"
    ```

- [Head Pose Estimation Model](https://docs.openvinotoolkit.org/latest/_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html)

    ```
    python <openvino directory>/deployment_tools/tools/model_downloader/downloader.py --name "landmarks-regression-retail-0009"
    ```

- [Gaze Estimation Model](https://docs.openvinotoolkit.org/latest/_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html)

    ```
    python <openvino directory>/deployment_tools/tools/model_downloader/downloader.py --name "gaze-estimation-adas-0002"
    ```

#### 3. Install Requirements

```
pip install -r requirements.txt
```

#### 4. Run the Application

```
python src/main.py -fd <openvino directory>/deployment_tools/open_model_zoo/tools/downloader/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001 -hp <openvino directory>/deployment_tools/open_model_zoo/tools/downloader/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001 -fl <openvino directory>/deployment_tools/open_model_zoo/tools/downloader/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009 -ge <openvino directory>/deployment_tools/open_model_zoo/tools/downloader/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002 -d CPU -i bin/demo.mp4 
```

## Demo

*TODO:* Explain how to run a basic demo of your model.

## Documentation

### Command Line Arguments

Argument|Type|Description
| ------------- | ------------- | -------------
-fd | Required | Path to .xml file of Face Detection model.
-fl | Required | Path to .xml file of Facial Landmark Detection model.
-hp| Required | Path to .xml file of Head Pose Estimation model.
-ge| Required | Path to .xml file of Gaze Estimation model.
-i| Required | Path to video file or enter cam for webcam
-probs  | Optional | Specify confidence threshold which the value here in range(0, 1), default=0.6
-flags | Optional | ff for faceDetectionModel, fl for landmarkRegressionModel, fh for headPoseEstimationModel, fg for gazeEstimationMode
-d | Optional | Provide the target device: CPU / GPU / VPU / FPGA

## Benchmarks

The Performance tests were run on HP Laptop with **Intel i7-8565U 1.99Ghz** and **16 GB Ram**

#### CPU

| Properties       | FP32        | FP16        | INT8        |
| ---------------- | ----------- | ----------- | ----------- |
| *Model Loading*  | 2.864384s   | 2.834568s   | 2.881543s   |
| *Inference Time* | 9.084232s   | 9.002353s   | 9.015497s   |
| *Total FPS*      | 1.245678fps | 2.67456fps | 2.134785fps |


## Results

As we can see from the results, models with lower precision gives us better inference time but they lose in accuracy. This happens as lower precision model uses less memory and they are less computationally expensive.
