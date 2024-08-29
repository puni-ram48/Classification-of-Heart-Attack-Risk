# 💔 Heart Attack Risk Classification Project

<img src="heart1.png" width="1000" height="466">

## Project Overview
Heart disease is a major cause of death around the world, so spotting people at risk early is really important for better health outcomes. This project uses machine learning to predict if someone might have a heart attack based on their health and personal details. The main goal is to create a model that helps both doctors and individuals assess heart attack risk, making it easier to take timely action and prevent potential issues.

## Dataset
The dataset used for this analysis is the "Heart Failure Prediction Dataset" by Fedesoriano, available on Kaggle. The dataset is accessible at [this link](https://www.kaggle.com/fedesoriano/heart-failure-prediction).

**Citation:**
Fedesoriano. (2021, September). *Heart failure prediction dataset*. Kaggle. Retrieved July 1, 2024, from https://www.kaggle.com/fedesoriano/heart-failure-prediction

- The dataset includes **918 observations** and **12 Features** in total.

- **Age:** Age of the patient in years.
- **Sex:** Gender of the patient (M for Male, F for Female).
- **ChestPainType:** Type of chest pain (TA for Typical Angina, ATA for Atypical Angina, NAP for Non-Anginal Pain, ASY for Asymptomatic).
- **RestingBP:** Resting blood pressure in mm Hg.
- **Cholesterol:** Serum cholesterol level in mg/dL.
- **FastingBS:** Fasting blood sugar level (1 if > 120 mg/dL, 0 otherwise).
- **RestingECG:** Results of resting electrocardiogram (Normal, ST for ST-T wave abnormality, LVH for left ventricular hypertrophy).
- **MaxHR:** Maximum heart rate achieved during exercise, a numeric value between 60 and 202.
- **ExerciseAngina:** Presence of exercise-induced angina (Y for Yes, N for No).
- **Oldpeak:** Depression of the ST segment during exercise, measured in numeric value.
- **ST_Slope:** Slope of the ST segment during peak exercise (Up for upsloping, Flat for flat, Down for downsloping).
- **HeartDisease:** Output class indicating heart disease (1 for Yes, 0 for No).

## Tools and Technologies Used
- **Data Analysis:** Python (Pandas, Numpy)
- **Machine Learning:** Scikit-Learn (Logistic Regression, Decision Tree, Random Forest, XGBoost, Gradient Boosting, ANN, K-Nearest Neighbors, SVM)
- **Visualization:** Matplotlib, Seaborn
- **Version Control:** Git, GitHub

## Installation and Usage
**Prerequisites**
Ensure Python is installed on your machine. You will also need to install the required libraries:

```bash
# Install dependencies
pip install -r requirements.txt
```

**Running the Project**
```bash
# Clone the repository
git clone https://github.com/puni-ram48/Heart-Attack-Classification.git
```
[**Data**](data): Contains the dataset for the project.

[**Project Analysis Report**](analysis_report.ipynb): Final report containing data analysis, visualizations, and model development details.

[**requirements.txt**](requirements.txt): List of required Python libraries.

## Model Development and Evaluation

**1. Data Preprocessing:**
   - Handled missing values and outliers.
   - Encoded categorical variables (e.g., 'Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope').
   - Scaled numerical features for improved model performance.

**2. Model Training:**

- **Trained Various Models:** Implemented and trained multiple machine learning algorithms, including Logistic Regression, Decision Trees, Random Forest, XGBoost, Gradient Boosting, Artificial Neural Networks (ANN), K-Nearest Neighbors (KNN), and Support Vector Machines (SVM).

- **Applied Cross-Validation:** Utilized 5-fold cross-validation to evaluate the performance of each model, ensuring reliable and consistent results.

- **Hyperparameter Tuning:** Optimized model performance using GridSearchCV for fine-tuning hyperparameters and enhancing model accuracy.

- **Used SMOTE:** Applied Synthetic Minority Over-sampling Technique (SMOTE) to address class imbalance by generating synthetic samples for the minority class, thereby improving model training and prediction accuracy.

 **3. Model Evaluation:**

- **Metrics Used:**
  - **Accuracy:** The ratio of correctly predicted instances to the total instances, indicating overall performance.
  - **ROC-AUC Score:** The area under the Receiver Operating Characteristic curve, which measures the model’s ability to distinguish between classes. This is crucial for imbalanced datasets.
  - **Recall:** The ability of the model to correctly identify positive cases, highlighting its effectiveness in detecting instances of heart disease.
  - **Precision:** The proportion of true positive predictions out of all positive predictions made, reflecting the accuracy of positive classifications.
  - **F1-Score:** The harmonic mean of precision and recall, providing a balanced measure of the model’s performance in terms of both precision and recall.

- **Best Model:**
  - **K-Nearest Neighbors (KNN)** was identified as the best model based on its high ROC-AUC score of 0.95 and excellent recall of 95%. This indicates that KNN is particularly effective at detecting heart attacks, making it the most suitable model for this project.

## Contributing
We welcome contributions to this project! If you would like to contribute, please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeature`).
3. Make your changes and commit them (`git commit -am 'Add some feature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Create a new Pull Request.

Please ensure your code is well-documented.

## Authors and Acknowledgment
This project was initiated and completed by Puneetha Dharmapura Shrirama. Special thanks to the Jeevitha DS for the guidance and support.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
