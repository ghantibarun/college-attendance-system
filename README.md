

![](https://github.com/p-p-p-p/college-attendance-system/blob/main/banner.png)

# College Attendance System
The "College Attendance System" is a powerful web application designed to effectively manage attendance records for both students and teachers in college courses. With this system, students and teachers can easily register for courses and the system will capture their name, ID, course name, and face embedding using a 30-second video from their webcam during registration.

The application offers several convenient features such as a tab to clean and prepare the dataset, another tab for building a support vector machine (SVM) model to accurately classify student/teacher faces, and a real-time attendance tab for taking attendance during classes.

The attendance records are stored in both sqlite and CSV formats, making it easy to view and analyze the data.With plans to integrate MongoDB in the future for efficient data storage. Additionally, the system provides an attendance analysis tab that allows users to review attendance data and generate reports.

By leveraging advanced technologies such as streamlit, single-shot detector (SSD)  and google facenet , the College Attendance System offers an accurate and efficient way to track attendance. With its automated attendance tracking capabilities, this system can help colleges comply with regulations, maintain accurate records, and improve overall efficiency.


This application is built using Python 3.10 .

## Run everything on google colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/p-p-p-p/college-attendance-system/blob/main/college_attendance_system.ipynb)


## Installation Instructions

### Step 1: Clone the Repository
First, clone this repository to your local machine using Git:
```bash
git clone https://github.com/p-p-p-p/college-attendance-system.git
```

### Step 2: Navigate to the Project Directory
Change into the project directory:
```bash
cd college-attendance-system/
```

### Step 3: Set Up a Virtual Environment
Create a virtual environment to manage dependencies:
```bash
python -m venv myenv
```

#### Activate the Virtual Environment
- **For Windows:**
  ```bash
  myenv\Scripts\activate
  ```
- **For Linux/Mac:**
  ```bash
  source myenv/bin/activate
  ```

### Step 4: Install Required Packages
Install the necessary dependencies from `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Step 5: Download the Model
Run the following command to download the required model:
```bash
python download_model.py
```

### Step 6: Run the Application
To start the application, execute:
```bash
streamlit run app.py
```


### Step 7: To Send Confirmation Email to Students and Attendance to Admin

1. Create an account at [Outlook Mail](https://www.microsoft.com/en-in/microsoft-365/outlook/email-and-calendar-software-microsoft-outlook).
2. Set the username and password in the `.env` file, and specify the admin email:

```plaintext
email="fakemail@outlook.com"
password="sadfkkjkj124845545496"
admin_mail="fake_Admin@gmail.com"
```

### Step 8: Deactivate and Clean Up
When you're done, deactivate the virtual environment and delete the project folder to clean up:
```
deactivate
cd ..
rmdir /s /q college-attendance-system
```
Or,
```
Just Delete the college-attendance-system Folder and clean temp folder on Windows
```

Delete all existing data and start with a clean codebase.
```
python clean.py
```
## Generate Unknown face class [unknow.csv](https://raw.githubusercontent.com/p-p-p-p/college-attendance-system/refs/heads/main/unknown.csv)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/p-p-p-p/college-attendance-system/blob/main/unknown_face.ipynb)




## Homepage
![](https://github.com/p-p-p-p/college-attendance-system/blob/main/images/homepage.png)
## Registration
![](https://github.com/p-p-p-p/college--attendance-system/blob/main/images/clip.gif)
![](https://github.com/p-p-p-p/college-attendance-system/blob/main/images/registration.png)
## Realtime attendance for every subject
![](https://github.com/p-p-p-p/college-attendance-system/blob/main/images/real.png)
## Sending confirmation mail in realtime
![](https://github.com/p-p-p-p/college-attendance-system/blob/main/images/confirmation.jpg)
## Clean dataset and train model
![](https://github.com/p-p-p-p/college--attendance-system/blob/main/images/4.png)

## After the completion of attendance, analyze all of the attendance records.
![](https://github.com/p-p-p-p/college--attendance-system/blob/main/images/5.png)
![](https://github.com/p-p-p-p/college-attendance-system/blob/main/images/ana.png)

## Send attendance records to teacher.
![](https://github.com/p-p-p-p/college-attendance-system/blob/main/images/send.png)
## Received Excel file by mail
![](https://github.com/p-p-p-p/college-attendance-system/blob/main/images/attend.jpg)
