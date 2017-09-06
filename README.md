# android-TFDetect

This sample demonstrates how to use [TensorFlow Android Inference Interface](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/android) and [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/object_detection/) to do object detection in android.

This project focuses on object detection, you may refer to https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android/ for other samples.


## Pre-requisites

- Android Studio

- Android SDK v21

- Android Build Tools v21.1.1

- Android Support Repository

- Camera2 API is not needed.

## Getting Started

1. Download the model and labels_list.

    ```bash
    # From android-TFDetect/assets/
    wget https://storage.googleapis.com/download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_android_export.zip
    unzip ssd_mobilenet_v1_android_export.zip
    rm ssd_mobilenet_v1_android_export.zip
    ```

2. Build.

    This sample uses the Gradle build system. To build this project, use the "gradlew build" command or use "Import Project" in Android Studio.
