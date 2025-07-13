# 🤖 ML Model Integration: scikit-learn to C++ (Eigen & ONNX Runtime)

This project serves as a proof-of-concept and a foundational component for a diploma thesis, demonstrating the integration of a machine learning model trained in Python's scikit-learn with a C++ application. The core idea is to leverage scikit-learn for rapid model prototyping and training, export the model to the **ONNX format**, and then perform efficient inference in a C++ environment using **ONNX Runtime** alongside the **Eigen mathematical framework** for numerical operations. This approach combines the flexibility of Python's ML ecosystem with the performance benefits of C++.

---

## 🧠 Core Concept & Features

The primary objective of this project is to bridge the gap between Python-based machine learning development and high-performance C++ application deployment. It showcases:

-   **Model Training in Python:** Demonstrates training a machine learning model (e.g., a Linear Regression model) using the `scikit-learn` library.
-   **ONNX Export:** Utilizes `skl2onnx` to convert the trained scikit-learn model into the interoperable **ONNX (Open Neural Network Exchange) format**. This is the crucial step for model portability.
-   **C++ Integration with ONNX Runtime & Eigen:** Implements the model's prediction logic in C++ by loading the ONNX model using the **ONNX Runtime** inference engine. The **Eigen library** is used for handling input/output data representation and any necessary pre/post-processing numerical computations.
-   **Data Flow:** Outlines the seamless process of passing input data to the C++ application, performing efficient inference using the ONNX model, and retrieving results.
-   **Specific Model Type:** This project specifically implements a ported **Linear Regression model** trained on simple numerical data, demonstrating the end-to-end workflow from Python to C++.

---

## 🖥️ Technologies Used

-   **Python 3.x**:
    -   `scikit-learn`: For machine learning model training.
    -   `NumPy`: Essential for numerical operations and data handling.
    -   `skl2onnx`: For converting scikit-learn models to ONNX format.
    -   `onnxruntime` (for Python testing): For verifying the ONNX model's inference in Python.
-   **C++**:
    -   **ONNX Runtime C++ API**: The inference engine used to load and run the ONNX model in C++.
    -   **Eigen 3.x**: A high-level C++ template library for linear algebra, matrix, and vector operations, used for data handling and any custom numerical logic.
    -   Standard C++ Libraries: For file I/O, data parsing, and general application logic.
-   **CMake** (Recommended for C++ projects): For managing the build process and linking with external libraries.

---

## 🚀 How to Run

### Part 1: Python - Model Training & ONNX Export

1.  **Install Python dependencies:**
    ```bash
    pip install scikit-learn numpy skl2onnx onnxruntime
    ```
2.  **Train the model and export to ONNX:**
    Run the Python script responsible for training your model and saving it as an `.onnx` file. This file will be consumed by the C++ application.
    ```bash
    python model.py
    ```
    This step will generate the `model.onnx` file, typically in a `model/` subdirectory as per the provided Python code.

3.  **Test ONNX model in Python (optional but recommended):**
    You can run `testing.py` to verify that the exported ONNX model performs as expected in a Python environment before porting to C++.
    ```bash
    python testing.py
    ```

### Part 2: C++ - ONNX Model Integration & Inference

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourUsername/YourProjectName.git](https://github.com/YourUsername/YourProjectName.git) # Replace with your actual repository URL
    cd YourProjectName # Navigate into the cloned directory
    ```

2.  **Ensure ONNX Runtime and Eigen are available:**
    You will need to set up the ONNX Runtime C++ API and the Eigen library for your C++ project.
    * **ONNX Runtime:** Download the pre-built binaries for your platform from the [official ONNX Runtime GitHub releases](https://github.com/microsoft/onnxruntime/releases) or build from source. You'll typically link against its libraries and include its headers.
    * **Eigen:** Install it via your system's package manager (e.g., `sudo apt-get install libeigen3-dev` on Debian/Ubuntu) or download directly from the [Eigen website](https://eigen.tuxfamily.org/index.php?title=Main_Page) and include its headers.

    *(Note: Your `test.cpp` uses direct paths like `D:\\dyplom\\praca\\onnxruntime-1.22.0\\include\\onnxruntime_cxx_api.h`. For a production-ready setup, these should be handled by a build system like CMake to ensure portability across different environments.)*

3.  **Build the C++ application:**
    It is highly recommended to use CMake for building C++ projects that depend on external libraries. Ensure your `CMakeLists.txt` correctly finds and links ONNX Runtime and Eigen.

    ```bash
    mkdir build
    cd build
    cmake ..
    make
    ```
    If you're compiling manually (less portable for complex projects):
    ```bash
    g++ -std=c++17 test.cpp -o your_application_name \
        -I/path/to/onnxruntime/include -L/path/to/onnxruntime/lib -lonnxruntime \
        -I/path/to/eigen # Adjust paths and library names as necessary
    ```

4.  **Place the ONNX model:**
    Ensure the `model/model.onnx` file (generated by `model.py`) is located where your C++ application expects it (e.g., in a `model/` subdirectory relative to the executable, or configure its path in the C++ code).

5.  **Run the C++ application:**
    ```bash
    ./your_application_name # Or ./test as per your current compilation output
    ```
    The C++ application will load the ONNX model, perform inference with example input, and output the predictions to the console.

### 🛠️ Prerequisites

-   Python 3.x
-   C++ Compiler (e.g., GCC/g++)
-   Python libraries: `scikit-learn`, `NumPy`, `skl2onnx`, `onnxruntime`
-   C++ libraries: **ONNX Runtime C++ API**, **Eigen 3.x**
-   (Optional but Recommended) `CMake` for building the C++ project.

---

## 🤝 Contribution & Contact

This project is part of a diploma thesis. For questions, collaboration opportunities, or feedback, please contact [Your Name/GitHub Username].
