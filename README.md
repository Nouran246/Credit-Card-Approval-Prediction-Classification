# Credit Card Approval Prediction (Classification Project)

**Authors:**
- [@Nouran Hassan](https://github.com/Nouran246)
- [@Roaa Khaled](https://github.com/Rowlkh)
- [@Malak Mohamed](https://github.com/MalakMohameed)
- [@Yahia-Elshobokshy](https://github.com/Yahia-Elshobokshy)

## About the Dataset:
Credit score cards are a common risk control method in the financial industry. They use personal information and data submitted by credit card applicants to predict the probability of future defaults and credit card borrowings. The bank uses this to decide whether to issue a credit card to the applicant. Credit scores objectively quantify the magnitude of risk.

**Goal:**
The goal of this project was to predict the credit card score status based on the applicant’s information, where the scores are mapped to either "Approved" or "Not Approved" based on certain criteria.

**Credit Scores to Predict:**
- `'C'`: "Credit Approved" status
- `'X'`: "Accepted" or "Approved" status
- `'0', '1', '2', '3', '4', '5'`: Levels of rejection, from minor to severe issues like insufficient credit history, high debt-to-income ratio, or bankruptcy.

### Mapping to Status:
- `'C'`: Approved
- `'X'`: Approved
- `'0', '1', '2', '3', '4', '5'`: Not Approved

The two datasets, `credit_record` and `application_record`, were merged based on the common column `ID` to combine applicant details with credit scores.

---

## Project Deliverables:
- A Python file with the complete implemented pipeline.
- A report explaining each step and decision in detail.

---

## Key Steps in the Project:

1. **Data Exploration and Statistical Analysis:**
   - We performed an initial exploration of both `credit_record` and `application_record` datasets.
   - Conducted sanity checks and determined the appropriate preprocessing techniques to ensure data quality.
   - Performed exploratory data analysis (EDA) to understand the distribution of features and identify any outliers, missing values, or discrepancies.

2. **Preprocessing:**
   - Checked for and handled any missing/null values in the datasets.
   - Applied feature engineering where necessary, including creating new features or transforming existing ones to improve model performance.
   - Dropped unnecessary columns that did not contribute to the prediction task.
   - Checked for class imbalance and addressed it through techniques like oversampling or undersampling.
   - Removed any duplicate records.
   - Merged the two datasets (`credit_record` and `application_record`) based on the `ID` column.
   - Applied label encoding to categorical features and performed feature scaling to ensure all features were on the same scale.

3. **Feature Selection Using Genetic Algorithms:**
   - Implemented Genetic Algorithms for feature selection to choose the most relevant features for model training.
   - The algorithm helped to determine which features contributed most to predicting the credit score status.

4. **Splitting the Data:**
   - Split the data into training (70%), validation (15%), and testing (15%) sets, ensuring that the data was well-distributed across the subsets for model training and evaluation.

5. **Model Training:**
   - We trained three classification models:
     - **K-Nearest Neighbors (KNN)**
     - **Decision Trees**
     - **Multi-Layer Perceptron (MLP)**
   - The models were trained using the selected features, and performance was evaluated using cross-validation on the validation set.

6. **Hyperparameter Tuning:**
   - Applied grid search to find the best hyperparameters for each model, tuning parameters such as the number of neighbors for KNN, the depth of decision trees, and the number of hidden layers for MLP.
   - This allowed us to optimize the models for better performance.

7. **Model Evaluation:**
   - Once hyperparameter tuning was completed, the models were evaluated using accuracy as the primary evaluation metric.
   - We found that the **Decision Trees** model performed the best in terms of accuracy, followed closely by **KNN**.

---

## Results:

- **Best Model:** Decision Trees
- **Evaluation Metric:** Accuracy

---

## Screenshots:
![Screenshot 2025-01-15 181348](https://github.com/user-attachments/assets/7d15ee8f-063d-4cf8-a31e-9dff6f1501d1)
![Screenshot 2025-01-15 181421](https://github.com/user-attachments/assets/1ecc4f79-ce5b-4583-b03e-c993688e2174)
![Screenshot 2025-01-15 181434](https://github.com/user-attachments/assets/67ec66f7-760e-4b16-8646-ea00211a895c)
![Screenshot 2025-01-15 181447](https://github.com/user-attachments/assets/fc89e4d9-0b19-4099-9373-6fe5eeb63074)

---

## Conclusion:
The project was successful in predicting the credit card approval status by leveraging classification models like KNN, Decision Trees, and MLP. Feature selection using Genetic Algorithms helped in improving model performance, and hyperparameter tuning ensured the optimal settings for each classifier. The Decision Trees model emerged as the best performer, achieving the highest accuracy in predicting whether a credit card application would be approved or not.
