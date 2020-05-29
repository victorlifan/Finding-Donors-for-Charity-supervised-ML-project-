>## [Finding Donors for CharityML enhanced](https://github.com/victorlifan/projects_review/tree/master/Finding%20Donors%20for%20CharityML)

## Project Title
Finding Donors for CharityML (supervised ML project)

## by Fan Li

## Date created
Project is created on April 23th 2020.


## Description
In this project, I employed several supervised algorithms of my choice to accurately model individuals' income using data collected from the 1994 U.S. Census. I then chose the best candidate algorithm from preliminary results and further optimize this algorithm to best model the data. My goal with this implementation is to construct a model that accurately predicts whether an individual makes more than $50,000. This sort of task can arise in a non-profit setting, where organizations survive on donations. Understanding an individual's income can help a non-profit better understand how large of a donation to request, or whether or not they should reach out to begin with. While it can be difficult to determine an individual's general income bracket directly from public sources, we can infer this value from other publically available features.

Workflow:
+ Exploring the Data
+ Preparing the Data
  - Transforming Skewed Continuous Features
  - Normalizing Numerical Features
  - 1-hot encoding
  - Shuffle and Split Data  
+ Evaluating Model Performance
  - Metrics and the Naive Predictor
  - Supervised Learning Models
  - Creating a Training and Predicting Pipeline
  - Initial Model Evaluation
+ Improving Results
  - Model Tuning
+ Feature Importance
  - Extracting Feature Importance
  - Feature Selection
  - Effects of Feature Selection


## Dataset

[`census.csv`](https://github.com/victorlifan/Finding-Donors-for-CharityML/blob/master/census.csv)

 The dataset for this project originates from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Census+Income). The datset was donated by Ron Kohavi and Barry Becker, after being published in the article "Scaling Up the Accuracy of Naive-Bayes Classifiers: A Decision-Tree Hybrid". You can find the article by Ron Kohavi [online](https://www.aaai.org/Papers/KDD/1996/KDD96-033.pdf). The data we investigate here consists of small changes to the original dataset, such as removing the 'fnlwgt' feature and records with missing or ill-formatted entries.

 ## Summary of Findings

 > Choosing the Best Model
 + F score would be the best evaluate indictor because our data is imbalanced. When 100% training data is used, we can tell from the comparisons above: AdaBoostClassifier has the height f_test score, which is 0.72. Besides, AdaBoost is the second fastest algorithm, but it is only slower than RandomForest which is the fastest one by less than couple second. Therefor, AdaBoost is the best option in this case.

 > Extracting Feature Importance
 + hours-per-week: working hour can highly reflect income
education_level/education-num: these two are highly correlated may reflect same information, which is the higher the education_level, the higher income
capital-gain/capital-gain: these two are highly negative correlated may reflect same information, which is the higher the capital-gain, the higher income
age: study shows age and worthiness is related

 > Effects of Feature Selection
 + both Accuracy and F-score dropped by using only 5 features. Accuracy is 3.05% less than best_predictions and F-score is 6.3% less.
when we look at the time consumed, use 5 features is 99.14% fatser than use all the features.
take all factors into consideration, use 5 key metrics is much less expensive, the small percentage of F-score dropped is a reasonable tradeoff.

 ## About
+ [`finding_donors.ipynb`](https://github.com/victorlifan/Finding-Donors-for-CharityML/blob/master/finding_donors.ipynb): This is the main file where I performed my work on the project.
+ [`census.csv`](https://github.com/victorlifan/Finding-Donors-for-CharityML/blob/master/census.csv): The project dataset.
+ [`visuals.py`](https://github.com/victorlifan/Finding-Donors-for-CharityML/blob/master/visuals.py): This Python script provides supplementary visualizations for the project. Do not modify.

## Software used
+ Jupyter Notebook
+ Python 3.7
> + Pandas
> + Numpy
> + Sklearn



## Credits
+ Data provided by:
    + [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Census+Income)
+ Instruction and assist: [Intro to Machine Learning with PyTorch](https://www.udacity.com/course/intro-to-machine-learning-nanodegree--nd229)
+ [Modern Machine Learning Algorithms: Strengths and Weaknesses](https://elitedatascience.com/machine-learning-algorithms)
+ [Choosing the Right Machine Learning Algorithm](https://hackernoon.com/choosing-the-right-machine-learning-algorithm-68126944ce1f)
