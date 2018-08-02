# OpenCV-Python Tutorial

----

## Contents

- [Introduction to OpenCV](#Introduction-to-OpenCV)
  - [Instruction to Install OpenCV-Python](#Instruction-to-Install-OpenCV-Python)
  - [Additional Resources](#Additional-Resources)
- [Image Processing in OpenCV](image-processing)
  - [Interactive Foreground Extraction using GrabCut Algorithm]()
- [Feature Detection and Description](feature-detection-and-description)
  - [Understanding Features](feature-detection-and-description/#Understanding-Features)  
  - [Harris Corner Detection](feature-detection-and-description/harris-corner-detection)
  - [Shi-Tomasi Corner Detector & Good Features to Track](feature-detection-and-description/shi-tomasi-detector)
  - [Introduction to SIFT (Scale-Invariant Feature Transform)](feature-detection-and-description/SIFT)
  - [Introduction to SURF (Speeded-Up Robust Features)](feature-detection-and-description/SURF)
  - [FAST Algorithm for Corner Detection](feature-detection-and-description/FAST)
  - [BRIEF (Binary Robust Independent Elementary Features)](feature-detection-and-description/BRIEF)
  - [ORB (Oriented FAST and Rotated BRIEF)](feature-detection-and-description/ORB)
  - [Feature Matching](feature-detection-and-description/feature-matching)
  - [Feature Matching + Homography to find Objects](feature-detection-and-description/homography)
- [Video Analysis](video-analysis)
  - [Meanshift and Camshift](video-analysis/meanshift-camshift)

## Introduction to OpenCV

### OpenCV

OpenCV was started at Intel in 1999 by Gary Bradsky, and the first release came out in 2000. Vadim Pisarevsky joined Gary Bradsky to manage Intel's Russian software OpenCV team. In 2005, OpenCV was used on Stanley, the vehicle that won the 2005 DARPA Grand Challenge. Later, its active development continued under the support of Willow Garage with Gary Bradsky and Vadim Pisarevsky leading the project. OpenCV now supports a multitude of algorithms related to Computer Vision and Machine Learning and is expanding day by day.

OpenCV supports a wide variety of programming languages such as C++, Python, Java, etc., and is available on different platforms including Windows, Linux, OS X, Android, and iOS. Interfaces for high-speed GPU operations based on CUDA and OpenCL are also under active development.

OpenCV-Python is the Python API for OpenCV, combining the best qualities of the OpenCV C++ API and the Python language.

### OpenCV-Python

OpenCV-Python is a library of Python bindings designed to solve computer vision problems.

Python is a general purpose programming language started by Guido van Rossum that became very popular very quickly, mainly because of its simplicity and code readability. It enables the programmer to express ideas in fewer lines of code without reducing readability.

Compared to languages like C/C++, Python is slower. That said, Python can be easily extended with C/C++, which allows us to write computationally intensive code in C/C++ and create Python wrappers that can be used as Python modules. This gives us two advantages: first, the code is as fast as the original C/C++ code (since it is the actual C++ code working in background) and second, it easier to code in Python than C/C++. OpenCV-Python is a Python wrapper for the original OpenCV C++ implementation.

OpenCV-Python makes use of Numpy, which is a highly optimized library for numerical operations with a MATLAB-style syntax. All the OpenCV array structures are converted to and from Numpy arrays. This also makes it easier to integrate with other libraries that use Numpy such as SciPy and Matplotlib.

### OpenCV-Python Tutorials

OpenCV introduces a new set of tutorials which will guide you through various functions available in OpenCV-Python. This guide is mainly focused on OpenCV 3.x version (although most of the tutorials will also work with OpenCV 2.x).

Prior knowledge of Python and Numpy is recommended as they won't be covered in this guide. Proficiency with Numpy is a must in order to write optimized code using OpenCV-Python.

### Instruction to Install OpenCV-Python

- [Windows](https://docs.opencv.org/3.4.1/d5/de5/tutorial_py_setup_in_windows.html)
- [Ubuntu](https://docs.opencv.org/3.4.1/d2/de6/tutorial_py_setup_in_ubuntu.html)

### Additional Resources
- [A Byte of Python](https://python.swaroopch.com/)
- [Basic Numpy Tutorials](http://scipy.github.io/old-wiki/pages/Tentative_NumPy_Tutorial)
- [Numpy Examples List](http://scipy.github.io/old-wiki/pages/Numpy_Example_List)
- [Matplotlib Tutorials](https://matplotlib.org/tutorials/index.html)
- [OpenCV Documentation](https://docs.opencv.org/)
- [OpenCV Forum](http://answers.opencv.org/questions/)
