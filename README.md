# BioMIDS
Project repository

**Team**:  BioMIDS
  

**Group Members**:  
Shusaku Asai, Saahithi Rao, Michael Tang   

**Data**:  
1. ICU Stays  
    https://physionet.org/content/mimiciii/1.4/ICUSTAYS.csv.gz  

2. Hospital Admissions (to determine patient death outcomes)  
    https://physionet.org/content/mimiciii/1.4/ADMISSIONS.csv.gz

3. More admissions information (i.e. procedures, prescriptions, and events)   
    https://physionet.org/content/mimiciii/1.4/INPUTEVENTS_CV.csv.gz  
    https://physionet.org/content/mimiciii/1.4/PROCEDUREEVENTS_MV.csv.gz  
    https://physionet.org/content/mimiciii/1.4/PRESCRIPTIONS.csv.gz  
  
  
**Objective**:  
The objective of the project is two-fold. We aim to 1) develop an algorithm that accurately identifies patients most at risk of death during an ICU visit, and 2) deploy the algorithm such that an end user inputs patient covariates into a dashboard that outputs probability of death.   
  
 **Data Science Plan**:  
 *Data plan (ETL)*:  The MIMIC III dataset is utilized to train, validate, and deploy the model. Due to the size of the full MIMIC III dataset, the demo data will first be downloaded as a proof-of-concept of the ETL, modeling, and deployment pipeline. The demo data is used to explore the data, construct feature engineering pipelines, and fit prediction models. The feature engineering and modeling pipelines will then be migrated to the entirety of the MIMIC III dataset, leveraging Google Cloud and BigQuery.  
Three primary datasets are used: ICU Stays, Hospital Admissions, and Procedural Events. The ICU Stays is used as the base table that provides information on patient ICU admission, discharge, length of stay, and unit location. The Hospital Admissions table is used to identify if the hospital stay resulted in “Death”. The Procedural Events table provides procedures performed on patients during each ICU stay.   
   
**ML Plan**:  
 - Select 3-5 classification models to train (e.g. Logistic Regression, Random Forest, XGBoost, Neural Network, etc. We aim to explore models with different complexity. Model training and validation will be performed solely on the training dataset).    
 - Determine the right metrics for performance evaluation (Since the data is slightly imbalanced, we will look into using F1 score, AUC,  Recall. Evaluation will be done on the training and test set)   
 - Each model hyperparameter will be finetuned based on selected evaluation metrics (hyperparameter tuning will be performed on the validation set)  
 - The best performing model will be used to construct a dashboard, where users could manually enter the values for each covariate/predictor. The dashboard is expected to output the likelihood of death during the ICU stay  
 - The dashboard will be deployed on AWS (Beanstalk is a host server that allows users to deploy Flask application)  
 
**Operations Plan (11 - 14 days):** 
- Data exploration (1 d)  
- Data transformation and merge (1 - 2 days)  
- Model selection (1  day )  
- Model training and validation (3 - 5 days)  
- Integration of model weights and dashboard (3 days)  
- Deployment on AWS (2 days)  

**Technology Stack:**  
We use Google BigQuery to access and query the full data. In order to do this, we connected a BigQuery account with our Physionet autenticated researcher account to gain access, following [the tutorial](https://mimic.mit.edu/docs/iii/tutorials/intro-to-mimic-iii-bq/).
SQL will be utilized to store the data as each table is ~6.2gb and join the necessary data into an analytic data frame. We will use several python packages to explore and analyze the data including: pandas and numpy (data manipulation), sklearn (logistic regression, random forest, etc.), matplotlib (visualizations). Lastly, we will create a dashboard using AWS to deploy our interactive model.   

 
 **Roles, Responsibilities and Timed Milestones** :    
*Shu*   
EDA part I - exploratory data analysis and visualizations (due: 11/10/21)  
EDA part II - feature engineering of hospital procedures of interest (due: 11/12/21)   
ML Part I - Fit and interpret first logistic regression and NN model (due: 11/15/21)  
ML Part II - Help Michael with hyper-parameter tuning of models (due: 11/16/21)  
Deployment Aid with Saahithi (due: 11/20/21)  

*Yi*   
Feature Engineering (possible dimension reduction) (due: 11/12/21)  
Machine Learning (Boosting model) (due: 11/15/21)  
Assist MT with model deployment (due: 11/21/21)  
 
*Saahithi*    
Query and merge data and explore data distributions, missingness etc (11/10/21)  
Help Yi and Shu with feature engineering and creating an analytic dataset (11/12/21)  
Build a ML (random forest model) and compare with models of team members (11/16/21)  

*Michael*   
Literature Review - look at existing methodologies on how experts handle EHR data (due: 11/11/21)  
Dashboard - integrate model within a Plotly dashboard (due: 11/19/21)  
Deployment - deploy model on AWS beanstalk (due: 11/21/21)  
  

