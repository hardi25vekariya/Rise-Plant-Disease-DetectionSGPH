# Rice Plant Disease Detection — Full Implementation Guide

This guide is for your **SGP Project — Phase 1**, which focuses only on **rice plants**.

Later, the same structure can be expanded to support multiple plants and multiple diseases.

## Project Flow

It is important to understand the overall flow:

```text
[1] Dataset (rice leaf images)
        |
        v
[2] Train CNN in Google Colab (train_model_colab.py)
        |
        v
[3] Download the trained model (rice_disease_model.h5 + labels.json)
        |
        v
[4] Place the model files inside the Flask web application's 'model/' folder
        |
        v
[5] Upload an image on the webpage → Flask uses the model to make a prediction
```

### Colab and the Webpage Are Two Different Things

**Google Colab** is used to train and create the machine learning model. This is generally a one-time process.

The **Flask webpage** uses the trained model to make predictions.

The `.h5` file connects these two parts. You download the trained `.h5` model from Google Colab and place it inside your Flask project.

---

# STEP-BY-STEP

## 1. Get the Dataset

Search for **"Rice Leaf Disease Dataset"** on Kaggle.

Download the dataset, create a ZIP file if required, and upload it to Google Drive.

Example path:

```text
MyDrive/rice_leaf_dataset.zip
```

---

## 2. Train the Model in Google Colab

Paste the complete `train_model_colab.py` code into a Google Colab notebook, cell by cell.

The code contains comments explaining each step.

Before starting the training:

**Runtime → Change runtime type → GPU**

Select **GPU** because training on a CPU can take significantly longer.

After training is completed, the following files will be generated/downloaded:

```text
rice_disease_model.h5
labels.json
```

---

## 3. Set Up the Local Flask Project

The project structure should look like this:

```text
rice_disease_project/
│
├── app.py                    ← Flask backend
├── requirements.txt
│
├── templates/
│   └── index.html            ← Webpage
│
├── static/
│   └── style.css             ← Website styling
│
└── model/                    ← Put the trained model files here
    ├── rice_disease_model.h5
    └── labels.json
```

Create the `model/` folder if it does not already exist.

Then place both files downloaded from Google Colab inside it:

```text
rice_disease_model.h5
labels.json
```

---

## 4. Install Dependencies and Run the Application

Open the terminal inside the project folder and run:

```bash
pip install -r requirements.txt
```

Then start the Flask application:

```bash
python app.py
```

Open the following address in your browser:

```text
http://127.0.0.1:5000
```

Upload a rice leaf image and click:

**"Check Disease"**

The application will display the predicted disease and confidence percentage.

---

# Important Tip for "Early Stage" Detection

The performance of the model depends heavily on the quality and variety of the training dataset.

If you specifically want to detect **early-stage symptoms**, the dataset should contain images showing mild or early symptoms.

Ideally, the dataset can contain:

* Images with mild or early-stage symptoms
* Images with more severe or advanced-stage symptoms

If the dataset provides sufficient stage-level information, you can create separate subclasses such as:

```text
Brown_spot_early
Brown_spot_advanced
```

However, if the available dataset does not provide reliable early/advanced labels, it is better to keep the project focused on **disease classification**.

This is still a valid and common SGP project scope.

For your report and viva, you can state:

> "Early symptom images were also included in the training dataset to help the model recognize mild disease symptoms."

Avoid claiming that the system specifically performs **early-stage disease detection** unless the model has been trained and evaluated specifically for that task.

---

# Deployment

If you want to make the webpage available online instead of running it only on your local computer, you can deploy the Flask application using a suitable hosting platform.

Possible platforms include:

* Render
* PythonAnywhere
* Railway

The general deployment process is:

```text
Local Flask Project
        ↓
Push Project to GitHub
        ↓
Connect GitHub Repository to Hosting Platform
        ↓
Configure Python Environment
        ↓
Deploy Flask Application
        ↓
Access Website Online
```

Make sure your deployment configuration includes all required dependencies from `requirements.txt`.

---

# Next Phase — Multi-Class / Multiple Plants

Once the rice-only version is working successfully, you can expand the project to support multiple plants.

For example:

```text
Rice
Tomato
Potato
Wheat
etc.
```

### 1. Add More Plant Datasets

Add datasets for additional plants and organize the disease images into appropriate folders.

### 2. Update the Training Dataset

The `DATASET_DIR` in `train_model_colab.py` can continue to point to the main dataset directory.

The training dataset can then contain disease folders for multiple plants.

The model will learn the available classes from the dataset structure.

### 3. Flask Application

If the Flask application is already designed to load class names dynamically from `labels.json`, the basic `app.py` and `index.html` structure can be reused.

The model and `labels.json` will contain the new classes after retraining.

Therefore, the current project structure provides a good foundation for future expansion.

---

# Final Project Flow

```text
                DATASET
                   ↓
          Google Colab Training
                   ↓
              CNN Model
                   ↓
       rice_disease_model.h5
                   +
              labels.json
                   ↓
             Flask Web App
                   ↓
           Upload Leaf Image
                   ↓
           Image Preprocessing
                   ↓
          CNN Model Prediction
                   ↓
       Disease + Confidence %
                   ↓
      Symptoms / Recommendation
```

**Phase 1:** Rice Plant Disease Detection

**Future Phase:** Multiple Plant Disease Detection
