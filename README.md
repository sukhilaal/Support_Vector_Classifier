# Support_Vector_Classifier
Implementation of Support Vector Machines (SVM) in Python using Scikit-Learn. Covers Support Vector Classification (SVC) on synthetic binary data with exploratory visualization, confusion matrix analysis, and GridSearchCV hyperparameter tuning, alongside synthetic dataset generation for Support Vector Regression (SVR).

# Support Vector Machines (SVC & SVR) Implementation

A practical machine learning notebook demonstrating **Support Vector Machines (SVM)** using Python and Scikit-Learn. The project walks through building, evaluating, and tuning a **Support Vector Classifier (SVC)** on synthetic binary classification data, followed by synthetic regression dataset preparation for **Support Vector Regression (SVR)**.

---

## 📋 Overview & Workflow

### 1. Support Vector Classification (SVC)
* **Synthetic Data Generation:** Generates a balanced binary classification dataset using `sklearn.datasets.make_classification` with:
  * Samples: $1,000$
  * Features: $2$ (`feature_0`, `feature_1`)
  * Classes: $2$ (`0` and `1`), with $2$ clusters per class and $0$ redundant features
* **Exploratory Data Analysis (EDA):** Structures the features and target labels into a Pandas DataFrame and visualizes class distributions using a Seaborn scatter plot (`sns.scatterplot`).
* **Train-Test Split:** Partitions the data into $70\%$ training ($700$ samples) and $30\%$ testing ($300$ samples) sets (`random_state=1`).
* **Baseline Model Training:** Fits a linear Support Vector Classifier (`SVC(kernel='linear')`) and inspects the learned hyperplane feature weights (`clf.coef_` $\approx [-0.1849, 1.7328]$).
* **Model Evaluation:** Evaluates predictions on the test set using accuracy score and both raw and normalized confusion matrices.
* **Hyperparameter Tuning:** Performs exhaustive $5$-fold cross-validation (`GridSearchCV`) across $40$ parameter combinations ($200$ total fits):
  * `C`: `[0.1, 0.2, 1, 2, 3, 10, 50, 100]`
  * `gamma`: `[1, 0.1, 0.2, 0.001, 0.003]`
  * `kernel`: `['linear']`

### 2. Support Vector Regression (SVR) Setup
* **Synthetic Regression Data:** Initializes a continuous target dataset using `sklearn.datasets.make_regression` with $1,000$ samples, $2$ features, $1$ target variable, and Gaussian noise ($\sigma = 3$) ready for SVR modeling.

---

## 📊 SVC Evaluation Results

| Metric / Parameter | Result |
| :--- | :--- |
| **Baseline Test Accuracy** | $90.0\%$ ($0.90$) |
| **Confusion Matrix (Raw)** | `[[122, 26], [4, 148]]` |
| **Class 0 Recall / False Positive Rate** | $82.43\%$ / $17.57\%$ |
| **Class 1 Recall / False Negative Rate** | $97.37\%$ / $2.63\%$ |
| **Best `GridSearchCV` Parameters** | `{'C': 2, 'gamma': 1, 'kernel': 'linear'}` |
| **Best $5$-Fold CV Accuracy** | $89.0\%$ ($0.89$) |

---

## 🛠️ Tech Stack & Dependencies

* **Python 3.x**
* **NumPy** & **Pandas** — Data manipulation and array operations
* **Matplotlib** & **Seaborn** — Data visualization
* **Scikit-Learn** — Synthetic dataset generation, preprocessing, `SVC`, `GridSearchCV`, and evaluation metrics

---

## 🚀 Getting Started

1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd <repository-folder>
   ```

2. **Install required packages:**
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn jupyter
   ```

3. **Run the notebook:**
   Launch Jupyter Notebook or Google Colab and execute the cells sequentially to reproduce the visualizations, model metrics, and grid search results.

   <img width="558" height="422" alt="image" src="https://github.com/user-attachments/assets/e18a9c9c-01e8-4a4c-8228-4a2aefe65095" />

   

