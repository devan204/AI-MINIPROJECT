# AI-MINIPROJECT
# Ex.No: 13 Learning – Use Supervised Learning(Miniproject)
## DATE: 07/05/2025
## REGISTER NUMBER : 212222060037
## AIM: 
To write a program to train the classifier for Student Exam Performance Prediction.
##  Algorithm:
Step 1 : Start the program.<br>
Step 2 : Import the necessary packages, such as NumPy, Pandas, Matplotlib, and Seaborn.<br>
Step 3 : Install and import Gradio for creating the user interface.<br>
Step 4 : Load the student performance dataset using Pandas.<br>
Step 5 : Perform exploratory data analysis (EDA) and visualize the data (optional).<br>
Step 6 : Check for missing values and handle them if necessary.<br>
Step 7 : Encode categorical features using LabelEncoder or OneHotEncoder.<br>
Step 8 : Split the dataset into input features (X) and target labels (y).<br>
Step 9 : Split the data into training and testing sets using train_test_split.<br>
Step 10 : Standardize the training and testing data using StandardScaler.<br>
Step 11 : Instantiate the MLPClassifier model with 1000 iterations and train the model on the training data.<br>
Step 12 : Print the model's accuracy on both the training and testing sets.<br>
Step 13 : Create a Gradio interface to take input values for student features and predict the exam performance using the trained model.<br>
Step 14 : Launch the Gradio interface for user interaction.<br>
Step 15 : Stop the program.<br>

## Program:

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import seaborn as sns

import matplotlib.pyplot as plt

# Load the dataset

data = pd.read_csv('/content/student_exam_data.csv - Sheet1.csv')

# Convert 'Pass/Fail' to numeric

label_encoder = LabelEncoder()

data['Pass/Fail'] = label_encoder.fit_transform(data['Pass/Fail'])  # Pass → 1, Fail → 0

X = data.drop('Pass/Fail', axis=1)

y = data['Pass/Fail']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(criterion='entropy', random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")

print(classification_report(y_test, y_pred, target_names=['Fail', 'Pass']))

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fail', 'Pass'], yticklabels=['Fail', 'Pass'])

plt.xlabel('Predicted')

plt.ylabel('Actual')

plt.title('Confusion Matrix')

plt.show()

from sklearn.tree import plot_tree

plt.figure(figsize=(16, 8))

plot_tree(model, feature_names=X.columns, class_names=['Fail', 'Pass'], filled=True)

plt.title("Decision Tree Visualization")

plt.show()

### Output:

![Screenshot (60)](https://github.com/user-attachments/assets/c86d7c7a-7a51-46ec-9270-b6b0dc96423d)

![Screenshot (61)](https://github.com/user-attachments/assets/02ec2a48-2790-42cf-9131-940fe4590db3)

![download](https://github.com/user-attachments/assets/54892e47-551f-4dd6-9604-16fc36271e18)

![download (1)2](https://github.com/user-attachments/assets/c18d9046-c347-43e2-813a-df4fcc98a9f4)

### Result:
Thus the system was trained successfully and the prediction was carried out.

