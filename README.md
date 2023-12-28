# Implementation of Multivariate Linear Regression
## Aim
To write a python program to implement multivariate linear regression and predict the output.
## Equipment’s required:
1.	Hardware – PCs
2.	Anaconda – Python 3.7 Installation / Moodle-Code Runner
## Algorithm:
### Step1
import pandas as pd
### Step2
read the csv file
### Step3
get the values of x and y variables
### Step4
create the linear regression model and fit
### Step5
print the predicted output.
## Program:
```
# DEVELOPED BY : THILAKESHWARAN
# REFERANCE NUMBER : 23013560
from sklearn.linear_model import LinearRegression
import numpy as np

X = np.array([[1, 2], [2, 3], [3, 4]])
y = np.array([3, 5, 7])

model = LinearRegression()
model.fit(X, y)

new_data = np.array([[4, 5]])
predicted_output = model.predict(new_data)

print(f'Predicted Output: {predicted_output}')

```
## Output:
![OUTPUT](https://github.com/Thilakeshwaran/Multivariate-Linear-Regression/assets/147473132/6e94aee2-9619-4fb9-b1e9-a87a2e20116c)

## Result
Thus the multivariate linear regression is implemented and predicted the output using python program.
