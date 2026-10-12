\# IMDb Movie Review Sentiment Analysis



\## 1. Project Overview



This project uses machine learning and natural language processing techniques to classify IMDb movie reviews as positive or negative.



The system converts textual reviews into numerical features using TF-IDF and trains machine learning classifiers to predict sentiment.



\## 2. Objectives



\- Process and analyze IMDb movie reviews.

\- Convert text into numerical representations using TF-IDF.

\- Train and compare multiple machine learning classifiers.

\- Evaluate model performance using classification metrics.

\- Predict sentiment for new movie reviews.

\- Visualize and compare model performance.



\## 3. Dataset



The project uses the IMDb Large Movie Review Dataset from Stanford University.



The dataset contains 50,000 labeled movie reviews:

\- 25,000 training reviews.

\- 25,000 test reviews.

\- Positive and negative sentiment labels.



Dataset source: https://ai.stanford.edu/\~amaas/data/sentiment/



The dataset must be downloaded and extracted locally before running the scripts. It is excluded from this repository.



\## 4. Technologies Used



\- Python

\- Pandas and NumPy

\- Scikit-learn

\- TF-IDF Vectorization

\- Logistic Regression

\- Multinomial Naive Bayes

\- Linear Support Vector Machine (Linear SVM)

\- Matplotlib and Seaborn



\## 5. Project Structure



```text

IMDb-Sentiment-Analysis/

├── main.py

├── predict.py

├── compare\_models.py

├── visualize\_results.py

├── requirements.txt

├── README.md

├── .gitignore

├── outputs/

│   ├── model\_comparison.csv

│   ├── model\_accuracy.png

│   ├── model\_f1\_score.png

│   └── confusion\_matrix.png

├── models/

└── aclImdb/

```



The dataset, virtual environment, and saved model files are excluded from version control.



\## 6. Installation



Install Python 3.12 or another compatible Python 3 version.



Clone the repository, enter the project directory, and create a virtual environment:



```powershell

python -m venv .venv

.\\.venv\\Scripts\\Activate.ps1

python -m pip install -r requirements.txt

```



Download the IMDb dataset from the source link above and extract it into the project directory so that the `aclImdb` folder is available.



\## 7. Running the Project



Train the Logistic Regression model:



```powershell

python main.py

```



Predict the sentiment of a new review:



```powershell

python predict.py

```



Compare the three classifiers:



```powershell

python compare\_models.py

```



Generate evaluation charts:



```powershell

python visualize\_results.py

```



Run the model comparison script before generating the visualization files.



\## 8. Evaluation



The models are evaluated using:

\- Accuracy

\- Precision

\- Recall

\- F1-score

\- Confusion matrix



The comparison results are saved in `outputs/model\_comparison.csv`. Charts are saved in the `outputs` directory.



Actual performance values should be taken from the program output.



\## 9. Conclusion



This project demonstrates how machine learning techniques can classify movie review sentiment using text preprocessing and TF-IDF feature extraction. Comparing multiple classifiers helps identify which approach performs best on the selected evaluation metrics.



