# SPAD512-SCOS GUI

A Python-based graphical user interface for controlling the **Pi Imaging SPAD512S** single-photon avalanche diode (SPAD) camera and performing **time-gated Speckle Contrast Optical Spectroscopy (SCOS)** and **Speckle Autocorrelation Coherence Spectroscopy (SACS)** measurements.

The software integrates camera control, data acquisition, real-time visualization, and offline analysis into a single application, providing a complete workflow for photon-counting optical experiments.

This project was developed as a **Final Year Project** for the B.Sc. in Electrical Engineering at **Bar-Ilan University**.

---

## Features

- Control and configure the Pi Imaging **SPAD512S** camera
- Live photon-counting image visualization
- Time-gated image acquisition
- TPSF (Temporal Point Spread Function) measurements
- IRF (Instrument Response Function) measurements
- SCOS processing
- SACS processing (under development)
- ROI-based analysis
- Automatic experiment saving
- Offline analysis tools for previously acquired measurements
- Modular processing pipeline for future algorithm development

---

# GUI Overview

The application consists of a central control panel and five dedicated analysis pages.

| Page | Description |
|------|-------------|
| **Live View** | Live visualization of incoming camera frames and camera operation. |
| **TPSF** | Acquisition and visualization of Temporal Point Spread Functions. |
| **IRF** | Instrument Response Function acquisition and visualization. |
| **SCOS** | Speckle Contrast Optical Spectroscopy acquisition and processing. |
| **SACS** | Speckle Autocorrelation Spectroscopy processing and visualization. |

Navigation between pages is handled through a stacked Qt interface while sharing the same camera connection and acquisition backend.

---

# Repository Structure

```
SPAD512-SCOS_GUI/

│
├── main.py
│   Application entry point.
│
├── camera_interface.py
│   High-level application logic coordinating the GUI,
│   camera operation and processing routines.
│
├── camera_wrapper.py
│   Wrapper around the Pi Imaging SPAD512S SDK used for
│   communicating with the camera.
│
├── config.json
│   Default acquisition and application configuration.
│
├── widgets/
│   Graphical user interface components.
│
│   ├── controls_panel.py
│   ├── live_view.py
│   ├── trace_view.py
│   ├── scos_view.py
│   ├── sacs_view.py
│   └── canvas.py
│
├── utils/
│   Core numerical algorithms.
│
│   ├── SCOS_calculation.py
│   ├── SACS_Calculations.py
│   ├── TPSF_calculation.py
│   ├── POS_calculation.py
│   └── data_handlers.py
│
├── scos_analysis/
│   Modular analysis package used for offline processing.
│
│   ├── data_loader.py
│   ├── signal_processing.py
│   ├── methods.py
│   └── visualization.py
│
└── Open_Results/
    Standalone analysis scripts and Jupyter notebooks
    for previously acquired datasets.
```

---

# Software Architecture

The project is organized into four logical layers.

## 1. Camera Layer

Responsible for communication with the Pi Imaging SPAD512S camera.

Main files:

- `camera_wrapper.py`
- `camera_interface.py`

Responsibilities include:

- Camera connection
- Camera configuration
- Time-gated acquisition
- Background worker threads
- Experiment management
- Communication with the GUI

---

## 2. Processing Layer

Implements the numerical algorithms used throughout the project.

Located primarily in

```
utils/
```

Current processing modules include

- SCOS calculations
- SACS calculations
- TPSF processing
- POS calculations
- Data loading and handling utilities

These modules are independent from the graphical interface, allowing them to be reused for offline analysis.

---

## 3. User Interface

The GUI is implemented using **PySide6 (Qt)**.

Each major experiment type is implemented as an independent widget.

```
widgets/
```

The main window combines

- Controls Panel
- Navigation
- Acquisition pages
- Plotting widgets
- Live visualization

This modular design makes it straightforward to add new experiment types without modifying the entire application.

---

## 4. Offline Analysis

The repository also contains tools for post-processing recorded experiments.

```
Open_Results/
scos_analysis/
```

These modules include:

- Data loading
- Signal processing
- Visualization
- Sensitivity analysis
- Interactive notebooks for result exploration

This separation allows experiments to be analyzed without requiring camera hardware.

---

# Processing Workflow

A typical SCOS experiment follows the workflow below.

```
SPAD512S Camera
        │
        ▼
Camera Configuration
        │
        ▼
Time-Gated Acquisition
        │
        ▼
Frame Collection
        │
        ▼
Processing Algorithms
        │
        ├── TPSF
        ├── IRF
        ├── SCOS
        └── SACS
        │
        ▼
Visualization
        │
        ▼
Save Results
        │
        ▼
Offline Analysis
```

---

# Dependencies

The project is written in **Python** and uses the following major libraries:

- PySide6
- NumPy
- SciPy
- Matplotlib
- OpenCV
- attrs

In addition, the project requires the **Pi Imaging SPAD512S software/SDK**, which provides communication with the camera hardware through the `SPAD512S` Python interface.

---

# Running the Software

Clone the repository

```bash
git clone https://github.com/ShiraTzi/SPAD512-SCOS_GUI.git
cd SPAD512-SCOS_GUI
```

After installing the required Python packages and the Pi Imaging SPAD512S software, launch the application with

```bash
python main.py
```

---

# Offline Analysis

Previously acquired datasets can be analyzed independently of the GUI using the tools provided in

```
Open_Results/
```

or by importing the reusable modules located in

```
scos_analysis/
```

These utilities provide a convenient environment for post-processing and visualization of experimental measurements.

---

# Future Development

The modular architecture of the project allows additional measurement techniques and processing algorithms to be integrated with minimal modifications to the existing codebase.

Potential future extensions include:

- Additional acquisition modes
- Automated calibration procedures
- Real-time quantitative analysis
- Improved experiment management
- Support for future SPAD camera models

---

# Acknowledgments

Developed as a fourth-year Electrical Engineering Final Project at **Bar-Ilan University**.

This software was created to support research in time-resolved Speckle Contrast Oprical Spectroscopy using the Pi Imaging SPAD512S camera.
