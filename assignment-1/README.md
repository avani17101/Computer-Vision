[![Generic badge](https://img.shields.io/badge/CV-Assignment:1-BLUE.svg)](https://shields.io/)
[![Generic badge](https://img.shields.io/badge/DUE-23:59hrs,03/02/2021-RED.svg)](https://shields.io/)
# Assignment-1
The goal of the assignment is to familiarize you to the process of camera calibration
and the critical role it plays in using any measurements of the world from images.

This assignment has been prepared by Meher Shashwat Nigam and Saraansh Tandon. Please raise doubts on the appropriate assignment thread on moodle.

# Instructions
- Follow the directory structure as shown below: 
  ```
  ├── src           
        ├── Assignment0.ipynb
  ├── images            //your images
  ├── calibration-data  //provided data
  └── README.md
  ```
- `src` will contain the Jupyter notebook(s) used for the assignment.
- `images` will contain images used for the questions.
- `calibration-data` contains images provided to you already, for solving the questions. 
- Follow this directory structure for all following assignments in this course.
- **Make sure you run your Jupyter notebook before committing, to save all outputs.**

## Helper code
Function to get Rotation matrix from Euler angles :
```
def eulerAnglesToRotationMatrix(theta) :
    R_x = np.array([[1,0,0],[0,math.cos(theta[0]),-math.sin(theta[0])],[0,math.sin(theta[0]), math.cos(theta[0])]])
    R_y = np.array([[math.cos(theta[1]),0,math.sin(theta[1])],[0,1,0],[-math.sin(theta[1]),0,math.cos(theta[1])]])             
    R_z = np.array([[math.cos(theta[2]),-math.sin(theta[2]),0],[math.sin(theta[2]),math.cos(theta[2]),0],[0,0,1]])
    R = np.dot(R_z,np.dot(R_y,R_x))
    return R
```
