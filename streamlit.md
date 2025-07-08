
## Streamlit Hydrogen Aircraft Certification Assistant Dashboard Setup Guide

This document provides step-by-step instructions for setting up and running the Hydrogen Aircraft Certification Assistant Dashboard on a new system. This dashboard is built using Streamlit and displays simulated data from FAA documents.

### 1. Prerequisites

Before you begin, ensure the following software is installed on your system:

-   **Python 3.8 or higher:** You can download it from [python.org](https://www.python.org/downloads/ "null"). Make sure to check the "Add Python to PATH" option during installation.
    
-   **pip (Python Package Installer):** This usually comes installed with Python.
    

### 2. File and Folder Setup

You need to prepare your project directory and the data files.

#### a. Create Your Project Folder

1.  Choose a location on your computer (e.g., your Desktop, Documents folder, or a dedicated `Projects` folder).
    
2.  Create a new folder named `streamlit_dashboard` (or any name you prefer). This will be your project's main directory.
    

#### b. Save the Python Code (`app.py`)

1.  You've already downloaded the `app.py` code.
    
2.  Place the `app.py` file directly inside the `streamlit_dashboard` folder you just created.
    

#### c. Prepare the PDF Text Files (`C:\pdfs`) - **CRUCIAL STEP**

The dashboard is configured to read `.txt` files from a specific location (`C:\pdfs`). You need to create this folder and populate it with dummy `.txt` files that match the names expected by the code.

1.  **Create the `pdfs` folder:**
    
    -   Go to your `C:\` drive (the root of your main hard drive).
        
    -   Create a new folder directly inside `C:\` named `pdfs`. The full path should be `C:\pdfs`.
        
2.  **Create the `.txt` files with exact names:** Inside your `C:\pdfs` folder, create the following empty text files. **The names must match EXACTLY, including capitalization, spaces, and the `.txt` extension.**
    
    -   `1. tc19-16 - Energy Supply Device ARC Recommendation Report.txt`
        
    -   `2. tc18-49 - Failure Mode and Effects Analysis on PEM Fuel Cell Systems for Aircraft Power Applications.txt`
        
    -   `tc17-23 - Flammability of Materials in a Low-Concentration Hydrogen Environment.txt`
        
    -   `tc21-3 - Fuel Tank Flammability Assessment Method User's Manual - Updated for Version 11.txt`
        
    -   `tc20-9 - Aircraft Fuel Cell and Safety Management System.txt`
        
    -   `tc19-55 - Aircraft Fuel Cell System.txt`
        
    -   `tc19-17 - Evaluation for a Lightweight Fuel Cell Containment System for Aircraft Safety.txt`
        
    -   `tc16-24 - Abusive Testing of Proton Exchange Membrane Hydrogen Fuel Cells.txt`
        
    -   `tc16-24.txt`
        
    -   `tc21-30 - Study of Unitized Regenerative Fuel Cell Systems for Aircraft Applications.txt`
        
    -   `tc19-16.txt`
        
    -   `tc18-49.txt`
        
    
    **How to create these files:**
    
    -   Right-click in the `C:\pdfs` folder, select `New` -> `Text Document`.
        
    -   Type the full name (e.g., `1. tc19-16 - Energy Supply Device ARC Recommendation Report.txt`) and press Enter.
        
    -   If Windows asks "Are you sure you want to change the file extension?", click "Yes".
        
    -   **Crucial:** To ensure you see the `.txt` extension, go to the "View" tab in File Explorer and make sure "File name extensions" is checked.
        
3.  **Add dummy content to each `.txt` file:** Open each `.txt` file you just created (e.g., by double-clicking it) and type a few sentences of dummy text into it. This content will be displayed in the dashboard. For example:
    
    -   `1. tc19-16 - Energy Supply Device ARC Recommendation Report.txt`: "This document discusses hydrogen fuel cell energy supply and related recommendations for aircraft. It covers safety aspects and temperature ranges."
        
    -   `2. tc18-49 - Failure Mode and Effects Analysis on PEM Fuel Cell Systems for Aircraft Power Applications.txt`: "This report details the Failure Mode and Effects Analysis (FMEA) for PEM fuel cell systems in aircraft. It focuses on identifying potential failures and their impact on power applications."
        
    -   ...and so on for all files.
        

### 3. Install Required Python Libraries

1.  Open your Command Prompt (Windows) or Terminal (macOS/Linux).
    
2.  Navigate to your `streamlit_dashboard` project folder using the `cd` command. For example: `cd C:\Users\YourName\streamlit_dashboard` (replace `YourName` with your actual username)
    
3.  Install Streamlit and Pandas by running this command: `pip install streamlit pandas`
    

### 4. Run the Streamlit Dashboard

1.  In the same Command Prompt/Terminal window (ensure you are still in your `streamlit_dashboard` folder), run the application: `streamlit run app.py`
    
2.  **Initial Prompt:** Streamlit might ask for your email address for updates. You can simply press `Enter` without typing anything to proceed.
    
3.  **Browser Launch:** Your default web browser should automatically open to the dashboard (usually at `http://localhost:8501`).
    
4.  **Manual Access (if browser doesn't open):** If the browser doesn't open, look in your terminal for a line like `Local URL: http://localhost:8501`. Copy this URL and paste it into your web browser's address bar, then press Enter.
    

### 5. Troubleshooting Common Issues

-   **`FileNotFoundError`:** This means a `.txt` file is missing or its name (including extension) doesn't exactly match what's in `app.py`. Double-check step 2c, especially showing file extensions in File Explorer.
    
-   **Browser Not Opening:** Refer to step 4.2 to manually open the `Local URL`.
    
-   **Stopping the App:** To stop the Streamlit app running in your terminal, go to the terminal window and press `Ctrl + C` (you might need to press it a couple of times).
    

You should now have a fully functional Streamlit dashboard running locally, displaying content from your specified `.txt` files!
