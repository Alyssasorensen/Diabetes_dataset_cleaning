# # HHA 550 Session 2 Homework Assignment

# ## Diabetes Dataset Cleaning

# **Context:** The dataset encompasses a decade (1999-2008) of clinical care data from 130 US hospitals and integrated delivery networks. Each entry pertains to the medical records of patients diagnosed with diabetes, covering laboratory tests, medications, and hospital stays of up to 14 days. The primary objective is to ascertain early readmission occurrences within 30 days post-discharge. This issue holds significance for several reasons. Despite compelling evidence demonstrating enhanced clinical outcomes with various preventive and therapeutic interventions for diabetic patients, a considerable number do not receive such treatments. This can be partially attributed to inconsistent diabetes management practices in hospital settings, neglecting glycemic control. Inadequate diabetes care not only escalates operational costs for hospitals due to patient readmissions but also adversely affects the morbidity and mortality of patients, exposing them to complications associated with diabetes.

# ## **Attribute Information:** 
# 
# **encounter_id:** unique identifier
# 
# **patient_nbr:** patient number
# 
# **race:** race of patient 
# 
# **gender:** "Female", "Male"
# 
# **age:** age of the patient 
# 
# **weight:** weight of the patient
# 
# **admission_type_id:** Integer identifier corresponding to 9 distinct values, for example, emergency, urgent, elective, newborn, and not available
# 
# **discharge_disposition_id:** Integer identifier corresponding to 29 distinct values, for example, discharged to home, expired, and not available
# 
# **admission_source_id:** Integer identifier corresponding to 21 distinct values, for example, physician referral, emergency room, and transfer from a hospital
# 
# **time_in_hospital:** Integer number of days between admission and discharge
# 
# **payer_code:** Integer identifier corresponding to 23 distinct values, for example, Blue Cross/Blue Shield, Medicare, and self-pay
# 
# **medical_specialty:** Integer identifier of a specialty of the admitting physician, corresponding to 84 distinct values, for example, cardiology, internal medicine, family/general practice, and surgeon
# 
# **num_lab_procedures:** Number of lab tests performed during the encounter
# 
# **num_procedures:** Number of procedures (other than lab tests) performed during the encounter
# 
# **num_medications:** Number of distinct generic names administered during the encounter
# 
# **number_outpatient:** Number of outpatient visits of the patient in the year preceding the encounter
# 
# **number_emergency:** Number of emergency visits of the patient in the year preceding the encounter
# 
# **number_inpatient:** Number of inpatient visits of the patient in the year preceding the encounter
# 
# **diag_1:** The primary diagnosis (coded as first three digits of ICD9); 848 distinct values
# 
# **diag_2:** Secondary diagnosis (coded as first three digits of ICD9); 923 distinct values
# 
# **diag_3:** Additional secondary diagnosis (coded as first three digits of ICD9); 954 distinct values
# 
# **number_diagnoses:** Number of diagnoses entered to the system
# 
# **max_glu_serum:** Indicates the range of the result or if the test was not taken. Values: >200, >300, normal, and none if not measured
# 
# **A1Cresult:** Indicates the range of the result or if the test was not taken. Values: >8 if the result was greater than 8%, >7 if the result was greater than 7% but less than 8%, normal if the result was less than 7%, and none if not measured.
# 
# **metformin:** The feature indicates whether the drug was prescribed or there was a change in the dosage. Values: up if the dosage was increased during the encounter, down if the dosage was decreased, steady if the dosage did not change, and no if the drug was not prescribed
# 
# **repaglinide:** The feature indicates whether the drug was prescribed or there was a change in the dosage. Values: up if the dosage was increased during the encounter, down if the dosage was decreased, steady if the dosage did not change, and no if the drug was not prescribed
# 
# **nateglinide:** The feature indicates whether the drug was prescribed or there was a change in the dosage. Values: up if the dosage was increased during the encounter, down if the dosage was decreased, steady if the dosage did not change, and no if the drug was not prescribed
# 
# **chlorpropamide:** The feature indicates whether the drug was prescribed or there was a change in the dosage. Values: up if the dosage was increased during the encounter, down if the dosage was decreased, steady if the dosage did not change, and no if the drug was not prescribed
# 
# **glimepiride:** The feature indicates whether the drug was prescribed or there was a change in the dosage. Values: up if the dosage was increased during the encounter, down if the dosage was decreased, steady if the dosage did not change, and no if the drug was not prescribed
# 
# **acetohexamide:** The feature indicates whether the drug was prescribed or there was a change in the dosage. Values: up if the dosage was increased during the encounter, down if the dosage was decreased, steady if the dosage did not change, and no if the drug was not prescribed
# 
# **glipizide:** The feature indicates whether the drug was prescribed or there was a change in the dosage. Values: up if the dosage was increased during the encounter, down if the dosage was decreased, steady if the dosage did not change, and no if the drug was not prescribed
# 
# **glyburide:** The feature indicates whether the drug was prescribed or there was a change in the dosage. Values: up if the dosage was increased during the encounter, down if the dosage was decreased, steady if the dosage did not change, and no if the drug was not prescribed
# 
# **tolbutamide:** The feature indicates whether the drug was prescribed or there was a change in the dosage. Values: up if the dosage was increased during the encounter, down if the dosage was decreased, steady if the dosage did not change, and no if the drug was not prescribed
# 
# **pioglitazone:** The feature indicates whether the drug was prescribed or there was a change in the dosage. Values: up if the dosage was increased during the encounter, down if the dosage was decreased, steady if the dosage did not change, and no if the drug was not prescribed
# 
# **rosiglitazone:** The feature indicates whether the drug was prescribed or there was a change in the dosage. Values: up if the dosage was increased during the encounter, down if the dosage was decreased, steady if the dosage did not change, and no if the drug was not prescribed
# 
# **acarbose:** The feature indicates whether the drug was prescribed or there was a change in the dosage. Values: up if the dosage was increased during the encounter, down if the dosage was decreased, steady if the dosage did not change, and no if the drug was not prescribed
# 
# **miglitol:** The feature indicates whether the drug was prescribed or there was a change in the dosage. Values: up if the dosage was increased during the encounter, down if the dosage was decreased, steady if the dosage did not change, and no if the drug was not prescribed
# 
# **troglitazone:** The feature indicates whether the drug was prescribed or there was a change in the dosage. Values: up if the dosage was increased during the encounter, down if the dosage was decreased, steady if the dosage did not change, and no if the drug was not prescribed
# 
# **tolazamide:** The feature indicates whether the drug was prescribed or there was a change in the dosage. Values: up if the dosage was increased during the encounter, down if the dosage was decreased, steady if the dosage did not change, and no if the drug was not prescribed
# 
# **examide:** The feature indicates whether the drug was prescribed or there was a change in the dosage. Values: up if the dosage was increased during the encounter, down if the dosage was decreased, steady if the dosage did not change, and no if the drug was not prescribed
# 
# **citoglipton:** The feature indicates whether the drug was prescribed or there was a change in the dosage. Values: up if the dosage was increased during the encounter, down if the dosage was decreased, steady if the dosage did not change, and no if the drug was not prescribed
# 
# **insulin:** The feature indicates whether the drug was prescribed or there was a change in the dosage. Values: up if the dosage was increased during the encounter, down if the dosage was decreased, steady if the dosage did not change, and no if the drug was not prescribed
# 
# **glyburide-metformin:** The feature indicates whether the drug was prescribed or there was a change in the dosage. Values: up if the dosage was increased during the encounter, down if the dosage was decreased, steady if the dosage did not change, and no if the drug was not prescribed
# 
# **glipizide-metformin:** The feature indicates whether the drug was prescribed or there was a change in the dosage. Values: up if the dosage was increased during the encounter, down if the dosage was decreased, steady if the dosage did not change, and no if the drug was not prescribed
# 
# **glimepiride-pioglitazone:** The feature indicates whether the drug was prescribed or there was a change in the dosage. Values: up if the dosage was increased during the encounter, down if the dosage was decreased, steady if the dosage did not change, and no if the drug was not prescribed
# 
# **metformin-rosiglitazone:** The feature indicates whether the drug was prescribed or there was a change in the dosage. Values: up if the dosage was increased during the encounter, down if the dosage was decreased, steady if the dosage did not change, and no if the drug was not prescribed
# 
# **metformin-pioglitazone:** The feature indicates whether the drug was prescribed or there was a change in the dosage. Values: up if the dosage was increased during the encounter, down if the dosage was decreased, steady if the dosage did not change, and no if the drug was not prescribed
# 
# **change:** Indicates if there was a change in diabetic medications (either dosage or generic name). Values: change and no change
# 
# **diabetesMed:** Indicates if there was any diabetic medication prescribed. Values: yes and no
# 
# **readmitted:** Days to inpatient readmission. Values: <30 if the patient was readmitted in less than 30 days, >30 if the patient was readmitted in more than 30 days, and No for no record of readmission.

# ## Install Libraries

# In[429]:


# Commands to install some of the libraries in-case if they are not installed
# Any other library that needs to be installed just use: !pip install <library name>

get_ipython().system('pip install seaborn')
get_ipython().system('pip install missingno')
get_ipython().system('pip install xgboost')
get_ipython().system('pip install catboost')
get_ipython().system('pip install regex')
get_ipython().system('pip install scikit-learn')
get_ipython().system('pip install pandas')
get_ipython().system('pip install numpy')
get_ipython().system('pip install imblearn')
get_ipython().system('pip install lightgbm')
# !pip install pyarrow">=14.0.1" --user
get_ipython().system('pip install --upgrade scikit-learn')


# ## Import Packages

# In[430]:


import pandas as pd   # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np   # linear algebra
import matplotlib.pyplot as plt  #graphs and plots
import seaborn as sns   #data visualizations 
import csv # Some extra functionalities for csv  files - reading it as a dictionary
from lightgbm import LGBMClassifier #sklearn is for machine learning and statistical modeling including classification, regression, clustering and dimensionality reduction 

from sklearn.model_selection import train_test_split, cross_validate   #break up dataset into train and test sets

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# importing python library for working with missing data
import missingno as msno
# To install missingno use: !pip install missingno
import re    # This library is used to perform regex pattern matching

# import various functions from sklearn
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from catboost import CatBoostClassifier
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, classification_report, make_scorer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split


# In[431]:


from sklearn.model_selection import KFold,cross_val_score, RepeatedStratifiedKFold,StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import OneHotEncoder,StandardScaler,PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer,SimpleImputer
from sklearn.compose import make_column_transformer
from imblearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, precision_score, recall_score, roc_auc_score, classification_report, f1_score
from sklearn.metrics import roc_curve, auc

# These won't work: from sklearn.metrics import plot_roc_curve (replaced with other one above) and plot_confusion_matrix


import plotly 
import plotly.express as px
import plotly.graph_objs as go
import plotly.offline as py
from plotly.offline import iplot
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

import warnings
warnings.filterwarnings("ignore")


# ## Loading in the diabetes dataset

# In[432]:


# Loading in the diabetes dataset

df = pd.read_csv('https://raw.githubusercontent.com/Alyssasorensen/Diabetes_dataset_cleaning/main/datasets/diabetic_data_final.csv')
df


# In[435]:


# Showing only the head data points on the dataset

df = pd.read_csv('https://raw.githubusercontent.com/Alyssasorensen/Diabetes_dataset_cleaning/main/datasets/diabetic_data_final.csv')
df.head()


# ## Exploring and Understanding the Data

# ### Initial Insights
# 
# * That makes it a lot easier to compare the missing value percentages within the columns.
# * In our dataset, we have both numerical and categorical variables.
# * It is essential to see whether columns are correctly inferred.
# * The most important one to look for is our target variable 'readmitted'
# * 'Readmitted' is detected as an object, not as an integer.
# * Target variable is coded as 1 for NO (was not readmitted), 2 for >30 (readmitted after 30 days), 3 for <30 (readmitted within 30 days)
# * The following are detected as integers: encounter_id, patient_nbr, admission_type_id, discharge_disposition_id, admission_source_id, time_in_hospital, num_lab_procedures, num_procedures, num_medications, number_outpatient, number_emergency, number_inpatient, number_diagnoses
# * In addition to these, we have 31 categorical variables, which we have to encode as numerical 

# ## What are the current problems we have that need to be solved?
# * We have a multi-class classification problem. 
# * We make prediction on the target variable `readmitted`
# * And we will build a model to get best prediction on the readmitted variable. 

# ## Insights into our target variable 
# * One of the first steps of exploratory data analysis should always be to look at what the values of y look like. 

# In[436]:


# Calculate the percentage of each category in the 'readmitted' column
readmitted_percentage = df['readmitted'].value_counts(normalize=True) * 100

# Display the result with similar formatting for three categories
print(f"Percentage of patients not readmitted: % {round(readmitted_percentage[0], 2)} --> ({df['readmitted'].value_counts()[0]} patients)")
print(f"Percentage of patients readmitted within <30 days: % {round(readmitted_percentage['<30'], 2)} --> ({df['readmitted'].value_counts()['<30']} patients)")
print(f"Percentage of patients readmitted after >30 days: % {round(readmitted_percentage['>30'], 2)} --> ({df['readmitted'].value_counts()['>30']} patients)")


# ## So what does that all mean?
# We have imbalanced data
# 
# * Almost 54% of the instances of our target variable are `not readmitted`
# * 54864 patients do not get readmitted
# * 11.16% of patients are readmitted `<30 days`
# * 34.93% of patients are readmitted `>30 days` 

# ## Visualize readmittance

# In[440]:


# Assuming 'readmitted' is a column in your DataFrame (df)
fig = px.histogram(df, x="readmitted", title='Readmitted with Diabetes', width=400, height=400)

# Show the plot
fig.show()


# In[339]:


# Assuming 'readmitted_percentage' is calculated
fig = px.bar(x=readmitted_percentage.index, y=readmitted_percentage, title='Readmitted with Diabetes', width=400, height=400, labels={'x': 'Readmitted Category', 'y': 'Percentage'})

# Show the plot
fig.show()


# * The readmitted for diabetes dataset is an example of a so-called imbalanced dataset.
# * The readmitted total is around 46%, which means that almost half of the patients were readmitted

# ## Data Imbalance
# 
# * Instances across classes are imbalanced, like in our dataset, we have imbalance data.
# 
# * The problem is, most of the machine learning algorithm do not work well with the imbalanced data.
# 
# * Some of the metrics (like accuracy) give us misleading results.
# 
# * Most of the time in classification problems our interest is to get better predict on the minority class.
# 
# * In our example: People that were readmitted before 30 days is minority class.
# 
# * In our example: Patients who did not readmit is majority class.

# In[340]:


df.info()


# ## Check for Missing Data / Missing Values

# In[341]:


# This is if we replace all "?" vaues with a 0

# List of columns to replace "?" with 0
columns_to_replace = ['race', 'weight', 'payer_code', 'medical_specialty', 'diag_1', 'diag_2', 'diag_3']

# Replace "?" with 0 in the specified columns
df[columns_to_replace] = df[columns_to_replace].replace('?', 0)
df


# In[342]:


# Turning race categorical values into numerical values 

# "0" is already defined as a missing value since executing the code from above, so we start off "1"  

# Define the mapping for the 'race' column
race_mapping = {'Caucasian': 1, 'AfricanAmerican': 2, 'Hispanic': 3}

# Map the 'race' column using the defined mapping
df['race'] = df['race'].map(race_mapping)

df


# In[343]:


# Turning gender categorical values into numerical values 

# "0" is already defined as a missing value since executing the code from above, so we start off "1"  

# Define the mapping for the 'gender' column
gender_mapping = {'Female': 1, 'Male': 2, 'Unknown/Invalid': 3}

df = df.replace({"gender": gender_mapping})
df


# In[344]:


# Turning age categorical values into numerical values 

# "0" is already defined as a missing value since executing the code from above, so we start off "1"  

# Define the mapping dictionary
age_mapping = {'[0-10)': 1, '[10-20)': 2, '[20-30)': 3, '[30-40)': 4, '[40-50)': 5, '[50-60)': 6, '[60-70)': 7, '[70-80)': 8, '[80-90)': 9, '[90-100)': 10}

df = df.replace({"age": age_mapping})
df


# In[345]:


# Turning payer_code categorical values into numerical values 

# "0" is already defined as a missing value since executing the code from above, so we start off "1"  

# Define the mapping for the 'payer_code' column
payer_code_mapping = {'MC': 1, 'MD': 2, 'SP': 3, 'CP': 4, 'UN': 5, 'HM': 6, 'BC': 7, 'SI': 8, 'DM': 9, 'CM': 10, 'PO': 11, 'WC': 12, 'OG': 13}

df = df.replace({"payer_code": payer_code_mapping})
df


# In[346]:


# NEED TO ADD MORE SPECIALTIES POSSIBLY

# Turning medical_specialty categorical values into numerical values 

# "0" is already defined as a missing value since executing the code from above, so we start off "1"  

# Define the mapping for the 'medical_specialty' column

medical_specialty_mapping = {'Pediatrics-Endocrinology': 1, 'InternalMedicine': 2, 'Family/GeneralPractice': 3, 'Cardiology': 4, 'Surgery-General': 5, 'Orthopedics': 6, 'Gastroenterology': 7, 'Nephrology': 8, 'Orthopedics-Reconstructive': 9, 'Surgery-Cardiovascular/Thoracic': 10, 'Psychiatry': 11, 'Emergency/Trauma': 12, 'Pulmonology': 13, 'Surgery-Neuro': 14, 'Obstetrics&Gynecology-GynecologicOnco': 15, 'Pediatrics': 16, 'ObstetricsandGynecology': 17, 'Hematology/Oncology': 18, 'Pediatrics-Endocrinology': 19, 'Otolaryngology': 20, 'Surgery-Colon&Rectal': 21, 'Urology': 22, 'Psychiatry-Child/Adolescent': 23, 'Gynecology': 24, 'Radiologist': 25, 'Surgery-Vascular': 26, 'PhysicalMedicineandRehabilitation': 27, 'Rheumatology': 28, 'Podiatry': 29, 'Hematology': 30, 'Osteopath': 31, 'Hospitalist': 32, 'Psychology': 33, 'InfectiousDiseases': 34, 'SportsMedicine': 35, 'Speech': 36, 'Perinatology': 37, 'Neurophysiology': 38, 'Pediatrics-InfectiousDiseases':39}

df = df.replace({"medical_specialty": medical_specialty_mapping})
df


# In[347]:


# Turning max_glu_serum categorical values into numerical values 

# # "0" is already defined as a missing value since executing the code from above, so we start off "1"  

# Replace NaN values in the 'max_glu_serum' column with a default value (e.g., 'None')
df['max_glu_serum'].fillna('None', inplace=True)

# Define the mapping for the 'max_glu_serum' column
max_glu_serum_mapping = {'None': 1, '>300': 2, '>200': 3, 'Norm': 4}

# Map values in the 'max_glu_serum' column using the defined mapping
df['max_glu_serum'] = df['max_glu_serum'].map(max_glu_serum_mapping)

# Display the resulting DataFrame
print(df['max_glu_serum'])


# In[348]:


# Turning A1Cresult categorical values into numerical values 

# "0" is already defined as a missing value since executing the code from above, so we start off "1"  

# Replace NaN values in the 'A1Cresult' column with a default value (e.g., 'None')
df['A1Cresult'].fillna('None', inplace=True)

# Define the mapping for the 'A1Cresult' column
A1Cresult_mapping = {'None': 1, '>7': 2, '>8': 3, 'Norm': 4}

# Map values in the 'A1Cresult' column using the defined mapping
df['A1Cresult'] = df['A1Cresult'].map(A1Cresult_mapping)

# Display the resulting DataFrame
print(df['A1Cresult'])


# In[349]:


# Turning metformin categorical values into numerical values 

# "0" is already defined as a missing value since executing the code from above, so we start off "1"  

# Define the mapping for the 'metformin' column
metformin_mapping = {'No': 1, 'Steady': 2, 'Up': 3, 'Down': 4, 'Yes': 5}

df = df.replace({"metformin": metformin_mapping})
df


# In[350]:


# Turning repaglinide categorical values into numerical values 

# "0" is already defined as a missing value since executing the code from above, so we start off "1"  

# Define the mapping for the 'repaglinide' column
repaglinide_mapping = {'No': 1, 'Steady': 2, 'Up': 3, 'Down': 4, 'Yes': 5}

df = df.replace({"repaglinide": repaglinide_mapping})
df


# In[351]:


# Turning nateglinide categorical values into numerical values 

# "0" is already defined as a missing value since executing the code from above, so we start off "1"  

# Define the mapping for the 'nateglinide' column
nateglinide_mapping = {'No': 1, 'Steady': 2, 'Up': 3, 'Down': 4, 'Yes': 5}

df = df.replace({"nateglinide": nateglinide_mapping})
df


# In[352]:


# Turning chlorpropamide categorical values into numerical values 

# "0" is already defined as a missing value since executing the code from above, so we start off "1"  

# Define the mapping for the 'chlorpropamide' column
chlorpropamide_mapping = {'No': 1, 'Steady': 2, 'Up': 3, 'Down': 4, 'Yes': 5}

df = df.replace({"chlorpropamide": chlorpropamide_mapping})
df


# In[353]:


# Turning glimepiride categorical values into numerical values 

# "0" is already defined as a missing value since executing the code from above, so we start off "1"  

# Define the mapping for the 'glimepiride' column
glimepiride_mapping = {'No': 1, 'Steady': 2, 'Up': 3, 'Down': 4, 'Yes': 5}

df = df.replace({"glimepiride": glimepiride_mapping})
df


# In[354]:


# Turning acetohexamide categorical values into numerical values 

# "0" is already defined as a missing value since executing the code from above, so we start off "1"  

# Define the mapping for the 'acetohexamide' column
acetohexamide_mapping = {'No': 1, 'Steady': 2, 'Up': 3, 'Down': 4, 'Yes': 5}

df = df.replace({"acetohexamide": acetohexamide_mapping})
df


# In[355]:


# Turning glipizide categorical values into numerical values 

# "0" is already defined as a missing value since executing the code from above, so we start off "1"  

# Define the mapping for the 'glipizide' column
glipizide_mapping = {'No': 1, 'Steady': 2, 'Up': 3, 'Down': 4, 'Yes': 5}

df = df.replace({"glipizide": glipizide_mapping})
df


# In[356]:


# Turning glyburide categorical values into numerical values 

# "0" is already defined as a missing value since executing the code from above, so we start off "1"  

# Define the mapping for the 'glyburide' column
glyburide_mapping = {'No': 1, 'Steady': 2, 'Up': 3, 'Down': 4, 'Yes': 5}

df = df.replace({"glyburide": glyburide_mapping})
df


# In[357]:


# Turning tolbutamide categorical values into numerical values 

# "0" is already defined as a missing value since executing the code from above, so we start off "1"  

# Define the mapping for the 'tolbutamide' column
tolbutamide_mapping = {'No': 1, 'Steady': 2, 'Up': 3, 'Down': 4, 'Yes': 5}

df = df.replace({"tolbutamide": tolbutamide_mapping})
df


# In[358]:


# Turning pioglitazone categorical values into numerical values 

# "0" is already defined as a missing value since executing the code from above, so we start off "1"  

# Define the mapping for the 'pioglitazone' column
pioglitazone_mapping = {'No': 1, 'Steady': 2, 'Up': 3, 'Down': 4, 'Yes': 5}

df = df.replace({"pioglitazone": pioglitazone_mapping})
df


# In[359]:


# Turning rosiglitazone categorical values into numerical values 

# "0" is already defined as a missing value since executing the code from above, so we start off "1"  

# Define the mapping for the 'rosiglitazone' column
rosiglitazone_mapping = {'No': 1, 'Steady': 2, 'Up': 3, 'Down': 4, 'Yes': 5}

df = df.replace({"rosiglitazone": rosiglitazone_mapping})
df


# In[360]:


# Turning acarbose categorical values into numerical values 

# "0" is already defined as a missing value since executing the code from above, so we start off "1"  

# Define the mapping for the 'acarbose' column
acarbose_mapping = {'No': 1, 'Steady': 2, 'Up': 3, 'Down': 4, 'Yes': 5}

df = df.replace({"acarbose": acarbose_mapping})
df


# In[361]:


# Turning miglitol categorical values into numerical values 

# "0" is already defined as a missing value since executing the code from above, so we start off "1"  

# Define the mapping for the 'miglitol' column
miglitol_mapping = {'No': 1, 'Steady': 2, 'Up': 3, 'Down': 4, 'Yes': 5}

df = df.replace({"miglitol": miglitol_mapping})
df


# In[362]:


# Turning troglitazone categorical values into numerical values 

# "0" is already defined as a missing value since executing the code from above, so we start off "1"  

# Define the mapping for the 'troglitazone' column
troglitazone_mapping = {'No': 1, 'Steady': 2, 'Up': 3, 'Down': 4, 'Yes': 5}

df = df.replace({"troglitazone": troglitazone_mapping})
df


# In[363]:


# Turning tolazamide categorical values into numerical values 

# "0" is already defined as a missing value since executing the code from above, so we start off "1"  

# Define the mapping for the 'tolazamide' column
tolazamide_mapping = {'No': 1, 'Steady': 2, 'Up': 3, 'Down': 4, 'Yes': 5}

df = df.replace({"tolazamide": tolazamide_mapping})
df


# In[364]:


# Turning examide categorical values into numerical values 

# "0" is already defined as a missing value since executing the code from above, so we start off "1"  

# Define the mapping for the 'examide' column
examide_mapping = {'No': 1, 'Steady': 2, 'Up': 3, 'Down': 4, 'Yes': 5}

df = df.replace({"examide": examide_mapping})
df


# In[365]:


# Turning citoglipton categorical values into numerical values 

# "0" is already defined as a missing value since executing the code from above, so we start off "1"  

# Define the mapping for the 'citoglipton' column
citoglipton_mapping = {'No': 1, 'Steady': 2, 'Up': 3, 'Down': 4, 'Yes': 5}

df = df.replace({"citoglipton": citoglipton_mapping})
df


# In[366]:


# Turning insulin categorical values into numerical values 

# "0" is already defined as a missing value since executing the code from above, so we start off "1"  

# Define the mapping for the 'insulin' column
insulin_mapping = {'No': 1, 'Steady': 2, 'Up': 3, 'Down': 4, 'Yes': 5}

df = df.replace({"insulin": insulin_mapping})
df


# In[367]:


# Turning glyburide-metformin categorical values into numerical values 

# "0" is already defined as a missing value since executing the code from above, so we start off "1"

# Define the mapping for the 'glyburide-metformin' column
glyburide_metformin_mapping = {'No': 1, 'Steady': 2, 'Up': 3, 'Down': 4, 'Yes': 5}

# Replace values in the 'glyburide-metformin' column using the defined mapping
df['glyburide-metformin'] = df['glyburide-metformin'].map(glyburide_metformin_mapping)

# Display the resulting DataFrame
print(df[['glyburide-metformin']])


# In[368]:


# Turning glipizide-metformin categorical values into numerical values 

# "0" is already defined as a missing value since executing the code from above, so we start off "1"

# Define the mapping for the 'glipizide-metformin' column
glipizide_metformin_mapping = {'No': 1, 'Steady': 2, 'Up': 3, 'Down': 4, 'Yes': 5}

# Replace values in the 'glipizide-metformin' column using the defined mapping
df['glipizide-metformin'] = df['glipizide-metformin'].map(glipizide_metformin_mapping)

# Display the resulting DataFrame
print(df[['glipizide-metformin']])


# In[369]:


# Turning glimepiride-pioglitazone categorical values into numerical values 

# "0" is already defined as a missing value since executing the code from above, so we start off "1"  

# Define the mapping for the 'glimepiride-pioglitazone' column
glimepiride_pioglitazone_mapping = {'No': 1, 'Steady': 2, 'Up': 3, 'Down': 4, 'Yes': 5}

# Replace values in the 'glimepiride-pioglitazone' column using the defined mapping
df['glimepiride-pioglitazone'] = df['glimepiride-pioglitazone'].map(glimepiride_pioglitazone_mapping)

# Display the resulting DataFrame
print(df[['glimepiride-pioglitazone']])


# In[370]:


# Turning metformin-rosiglitazone categorical values into numerical values 

# "0" is already defined as a missing value since executing the code from above, so we start off "1"  

# Define the mapping for the 'metformin-rosiglitazone' column
metformin_rosiglitazone_mapping = {'No': 1, 'Steady': 2, 'Up': 3, 'Down': 4, 'Yes': 5}

# Replace values in the 'metformin-rosiglitazone' column using the defined mapping
df['metformin-rosiglitazone'] = df['metformin-rosiglitazone'].map(metformin_rosiglitazone_mapping)

# Display the resulting DataFrame
print(df[['metformin-rosiglitazone']])


# In[371]:


# Turning metformin-pioglitazone categorical values into numerical values 

# "0" is already defined as a missing value since executing the code from above, so we start off "1"  

# Define the mapping for the 'metformin-pioglitazone' column
metformin_pioglitazone_mapping = {'No': 1, 'Steady': 2, 'Up': 3, 'Down': 4, 'Yes': 5}

# Replace values in the 'metformin-pioglitazone' column using the defined mapping
df['metformin-pioglitazone'] = df['metformin-pioglitazone'].map(metformin_pioglitazone_mapping)

# Display the resulting DataFrame
print(df[['metformin-pioglitazone']])


# In[372]:


# Turning change categorical values into numerical values 

# "0" is already defined as a missing value since executing the code from above, so we start off "1"  

# Define the mapping for the 'change' column
change_mapping = {'No': 1, 'Ch': 2, 'Yes': 3}

df = df.replace({"change": change_mapping})
df


# In[373]:


# Turning diabetesMed categorical values into numerical values 

# "0" is already defined as a missing value since executing the code from above, so we start off "1"  

# Define the mapping for the 'diabetesMed' column
diabetesMed_mapping = {'No': 1, 'Ch': 2, 'Yes': 3}

df = df.replace({"diabetesMed": diabetesMed_mapping})
df


# In[374]:


# Turning readmitted categorical values into numerical values 

# "0" is already defined as a missing value since executing the code from above, so we start off "1"  

# Define the mapping for the 'readmitted' column
readmitted_mapping = {'NO': 1, '>30': 2, '<30': 3}

df = df.replace({"readmitted": readmitted_mapping})
df


# In[375]:


def missing (df):
    missing_number = df.isnull().sum().sort_values(ascending=False)
    missing_percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    missing_values = pd.concat([missing_number, missing_percent], axis=1, keys=['Missing_Number', 'Missing_Percent'])
    return missing_values

missing(df)


# In[376]:


# Visualizing the missing data to get more idea
msno.bar(df)


# In[377]:


msno.matrix(df)


# * Missing values at race 
# * Rnadom missing values
# * Handle it by using pipeline during the modeling

# ## Numerical Features
# * Look at the data elements (columns) using `df.head()`
# * Look at the Dtype (data type) using `df.info()`

# In[378]:


df.head()


# In[379]:


df.info()


# In[380]:


print(df.columns)


# In[381]:


categorical = ['race', 'gender', 'age', 'weight', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id', 'payer_code', 'medical_specialty', 'diag_1', 'diag_2', 'diag_3', 'max_glu_serum', 'A1Cresult', 'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 'rosigitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone', 'change', 'diabetesMed', 'readmitted']

numerical = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications', 'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses']


# In[382]:


df[numerical].describe()


# * We have eight numerical features in our dataset
# * A majority of them are measured in different scales
# * Based on the mean and median score differences, we can expect:
# * Slight right skew on 'time_in_hospital'
# * Slight left skew on 'num_lab_procedures'
# * Slight right skew on 'num_procedures'
# * Slight right skew on 'num_medications'
# * Slight right skew on 'number_outpatient', 'number_emergency', and 'number_inpatient'; these three columns have a lot of zeroes which could be affecting the interpretation, but often right-skewed due to the number of zeroes
# * Slight left skew on 'number_diagnoses'

# ## Skewness

# In[383]:


df[numerical].skew()


# * Based on the results, time_in_hospital is moderately right-skewed (positive), num_lab_procedures is slightly left-skewed (negative), num_procedures is moderately right-skewed (positive), num_medications is moderately right-skewed (positive), number_outpatient is heavily right-skewed (positive), number_emergency is extremely right-skewed (positive), number_inpatient is highly right-skewed (positive), number_diagnoses is slightly left-skewed (negative)

# ## Univariate Analysis

# In[384]:


df[numerical].hist(figsize=(20,10));


# * As seen in both skewness result and histograms, numerical features have skewness in different degrees
# * We will deal with different scale and skewness during the modeling by using standardization `Standard scaler`

# ## Categorical Features
# * Race
# * Gender
# * Age 
# * Weight
# * Admission_type_id
# * Discharge_disposition_id 
# * Admission_source_id
# * Payer_code
# * Medical_specialty
# * Diag_1
# * Diag_2 
# * Diag_3 
# * Max_glu_serum
# * A1Cresult
# * Metformin
# * Repaglinide
# * Nateglinide
# * Chlorpropamide
# * Glimepiride
# * Acetohexamide 
# * Glipizide
# * Glyburide 
# * Tolbutamide
# * Pioglitazone 
# * Rosiglitazone 
# * Acarbose 
# * Miglitol
# * Troglitazone
# * Tolazamide
# * Examide 
# * Citoglipton
# * Insulin 
# * Glyburide-metformin
# * Glipizide-metformin 
# * Glimepiride-pioglitazone
# * Metformin-rosiglitazone
# * Metformin-pioglitazone
# * Change
# * DiabetesMed
# * Readmitted

# ## Race

# In[385]:


print (f'{round(df["race"].value_counts(normalize=True)*100,2)}')
fig = px.histogram(df, x="race", title='Race', width=750, height=750)
fig.update_layout(xaxis={'categoryorder':'total descending'})
fig.show()

#increase width and height for graphs with more than 2 choices
#try increasing to see what fits best


# ### Race Counts 
# * 76,099 Caucasian
# * 19,210 African American
# * 2,037 Hispanic

# ## Gender 

# In[386]:


print (f'{round(df["gender"].value_counts(normalize=True)*100,2)}')
fig = px.histogram(df, x="gender", title='Gender', width=750, height=750)
fig.update_layout(xaxis={'categoryorder':'total descending'})
fig.show()


# * 54,708 females
# * 47,055 males
# * 0 invalid/unknown

# ## Age

# In[387]:


print (f'{round(df["age"].value_counts(normalize=True)*100,2)}')
fig = px.histogram(df, x="age", title='Age', width=750, height=750)
fig.update_layout(xaxis={'categoryorder':'total descending'})
fig.show()


# * 161 (ages 0-10)
# * 691 (ages 10-20)
# * 1,657 (ages 20-30)
# * 3,775 (ages 30-40)
# * 9,685 (ages 40-50)
# * 17,256 (ages 50-60)
# * 22,483 (ages 60-70)
# * 26,068 (ages 70-80)
# * 17,197 (ages 80-90)
# * 2,793 (ages 90-100)

# ## Weight

# In[388]:


print (f'{round(df["weight"].value_counts(normalize=True)*100,2)}')
fig = px.histogram(df, x="weight", title='Weight', width=750, height=750)
fig.update_layout(xaxis={'categoryorder':'total descending'})
fig.show()


# * 98,569 unknown
# * 1336 (75-100)
# * 897 (50-75)
# * 0.05% (0-25)
# * 625 (100-125)
# * 0.14% (125-150)
# * 0.10% (25-50)
# * 0.03% (150-175)
# * 0.01% (175-200)
# * 0 (>200) 
# 
# 
# 
# 
# 
# 
# 

# ## Admission Type ID

# In[389]:


print (f'{round(df["admission_type_id"].value_counts(normalize=True)*100,2)}')
fig = px.histogram(df, x="admission_type_id", title='Admission Type ID', width=750, height=750)
fig.update_layout(xaxis={'categoryorder':'total descending'})
fig.show()


# * 53,990 1
# * 18,480 2
# * 18,869 3
# * 0.01% 4
# * 4,785 5
# * 5,291 6
# * 0.02% 7
# * 320 8

# ## Discharge Disposition ID

# In[390]:


print (f'{round(df["discharge_disposition_id"].value_counts(normalize=True)*100,2)}')
fig = px.histogram(df, x="discharge_disposition_id", title='Discharge Disposition ID', width=750, height=750)
fig.update_layout(xaxis={'categoryorder':'total descending'})
fig.show()


# * 60,234 1
# * 2,128 2
# * 13,954 3
# * 815 4
# * 1184 5
# * 12,902 6
# * 623 7
# * 0.11% 8
# * 0.02% 9
# * 0.01 10
# * 1642 11
# * 0 12
# * 0.39% 13
# * 0.37% 14
# * 0.06% 15
# * 0.01% 16
# * 0.01% 17
# * 3,691 18
# * 0.01% 19
# * 0 20
# * 0 21
# * 1,993 22
# * 412 23
# * 48 24
# * 989 25
# * 0 26
# * 0 27

# ## Admission Source ID

# In[391]:


print (f'{round(df["admission_source_id"].value_counts(normalize=True)*100,2)}')
fig = px.histogram(df, x="admission_source_id", title='Admission Source ID', width=750, height=750)
fig.update_layout(xaxis={'categoryorder':'total descending'})
fig.show()


# ## Payer Code 

# In[392]:


print (f'{round(df["payer_code"].value_counts(normalize=True)*100,2)}')
fig = px.histogram(df, x="payer_code", title='payer_code', width=750, height=750)
fig.update_layout(xaxis={'categoryorder':'total descending'})
fig.show()


# ## Medical Specialty

# In[393]:


print (f'{round(df["medical_specialty"].value_counts(normalize=True)*100,2)}')
fig = px.histogram(df, x="medical_specialty", title='Medical Specialty', width=750, height=750)
fig.update_layout(xaxis={'categoryorder':'total descending'})
fig.show()


# ## Diag 1

# In[394]:


print (f'{round(df["diag_1"].value_counts(normalize=True)*100,2)}')
fig = px.histogram(df, x="diag_1", title='diag_1', width=750, height=750)
fig.update_layout(xaxis={'categoryorder':'total descending'})
fig.show()


# ## Diag 2

# In[395]:


print (f'{round(df["diag_2"].value_counts(normalize=True)*100,2)}')
fig = px.histogram(df, x="diag_2", title='diag_2', width=750, height=750)
fig.update_layout(xaxis={'categoryorder':'total descending'})
fig.show()


# ## Diag 3

# In[396]:


print (f'{round(df["diag_3"].value_counts(normalize=True)*100,2)}')
fig = px.histogram(df, x="diag_3", title='diag_3', width=750, height=750)
fig.update_layout(xaxis={'categoryorder':'total descending'})
fig.show()


# ## Max_Glu_Serum

# In[397]:


print (f'{round(df["max_glu_serum"].value_counts(normalize=True)*100,2)}')
fig = px.histogram(df, x="max_glu_serum", title='max_glu_serum', width=750, height=750)
fig.update_layout(xaxis={'categoryorder':'total descending'})
fig.show()


# ## A1CResult

# In[398]:


print (f'{round(df["A1Cresult"].value_counts(normalize=True)*100,2)}')
fig = px.histogram(df, x="A1Cresult", title='A1Cresult', width=750, height=750)
fig.update_layout(xaxis={'categoryorder':'total descending'})
fig.show()


# ## Metformin

# In[399]:


print (f'{round(df["metformin"].value_counts(normalize=True)*100,2)}')
fig = px.histogram(df, x="metformin", title='metformin', width=750, height=750)
fig.update_layout(xaxis={'categoryorder':'total descending'})
fig.show()


# ## Repaglinide

# In[400]:


print (f'{round(df["repaglinide"].value_counts(normalize=True)*100,2)}')
fig = px.histogram(df, x="repaglinide", title='repaglinide', width=750, height=750)
fig.update_layout(xaxis={'categoryorder':'total descending'})
fig.show()


# ## Nateglinide

# In[401]:


print (f'{round(df["nateglinide"].value_counts(normalize=True)*100,2)}')
fig = px.histogram(df, x="nateglinide", title='nateglinide', width=750, height=750)
fig.update_layout(xaxis={'categoryorder':'total descending'})
fig.show()


# ## Chlorpropamide

# In[402]:


print (f'{round(df["chlorpropamide"].value_counts(normalize=True)*100,2)}')
fig = px.histogram(df, x="chlorpropamide", title='chlorpropamide', width=750, height=750)
fig.update_layout(xaxis={'categoryorder':'total descending'})
fig.show()


# ## Glimepiride

# In[403]:


print (f'{round(df["glimepiride"].value_counts(normalize=True)*100,2)}')
fig = px.histogram(df, x="glimepiride", title='glimepiride', width=750, height=750)
fig.update_layout(xaxis={'categoryorder':'total descending'})
fig.show()


# ## Acetohexamide

# In[404]:


print (f'{round(df["acetohexamide"].value_counts(normalize=True)*100,2)}')
fig = px.histogram(df, x="acetohexamide", title='acetohexamide', width=750, height=750)
fig.update_layout(xaxis={'categoryorder':'total descending'})
fig.show()


# ## Glipizide

# In[405]:


print (f'{round(df["glipizide"].value_counts(normalize=True)*100,2)}')
fig = px.histogram(df, x="glipizide", title='glipizide', width=750, height=750)
fig.update_layout(xaxis={'categoryorder':'total descending'})
fig.show()


# ## Glyburide

# In[406]:


print (f'{round(df["glyburide"].value_counts(normalize=True)*100,2)}')
fig = px.histogram(df, x="glyburide", title='glyburide', width=750, height=750)
fig.update_layout(xaxis={'categoryorder':'total descending'})
fig.show()


# ## Tolbutamide

# In[407]:


print (f'{round(df["tolbutamide"].value_counts(normalize=True)*100,2)}')
fig = px.histogram(df, x="tolbutamide", title='tolbutamide', width=750, height=750)
fig.update_layout(xaxis={'categoryorder':'total descending'})
fig.show()


# ## Pioglitazone

# In[408]:


print (f'{round(df["pioglitazone"].value_counts(normalize=True)*100,2)}')
fig = px.histogram(df, x="pioglitazone", title='pioglitazone', width=750, height=750)
fig.update_layout(xaxis={'categoryorder':'total descending'})
fig.show()


# ## Rosiglitazone

# In[409]:


print (f'{round(df["rosiglitazone"].value_counts(normalize=True)*100,2)}')
fig = px.histogram(df, x="rosiglitazone", title='rosiglitazone', width=750, height=750)
fig.update_layout(xaxis={'categoryorder':'total descending'})
fig.show()


# ## Acarbose

# In[ ]:


print (f'{round(df["acarbose"].value_counts(normalize=True)*100,2)}')
fig = px.histogram(df, x="acarbose", title='acarbose', width=750, height=750)
fig.update_layout(xaxis={'categoryorder':'total descending'})
fig.show()


# ## Miglitol

# In[313]:


print (f'{round(df["miglitol"].value_counts(normalize=True)*100,2)}')
fig = px.histogram(df, x="miglitol", title='miglitol', width=750, height=750)
fig.update_layout(xaxis={'categoryorder':'total descending'})
fig.show()


# ## Troglitazone

# In[314]:


print (f'{round(df["troglitazone"].value_counts(normalize=True)*100,2)}')
fig = px.histogram(df, x="troglitazone", title='troglitazone', width=750, height=750)
fig.update_layout(xaxis={'categoryorder':'total descending'})
fig.show()


# ## Tolazamide

# In[315]:


print (f'{round(df["tolazamide"].value_counts(normalize=True)*100,2)}')
fig = px.histogram(df, x="tolazamide", title='tolazamide', width=750, height=750)
fig.update_layout(xaxis={'categoryorder':'total descending'})
fig.show()


# ## Examide

# In[318]:


print (f'{round(df["examide"].value_counts(normalize=True)*100,2)}')
fig = px.histogram(df, x="examide", title='examide', width=300, height=750)
fig.update_layout(xaxis={'categoryorder':'total descending'})
fig.show()


# ## Citoglipton

# In[320]:


print (f'{round(df["citoglipton"].value_counts(normalize=True)*100,2)}')
fig = px.histogram(df, x="citoglipton", title='citoglipton', width=300, height=750)
fig.update_layout(xaxis={'categoryorder':'total descending'})
fig.show()


# ## Insulin

# In[322]:


print (f'{round(df["insulin"].value_counts(normalize=True)*100,2)}')
fig = px.histogram(df, x="insulin", title='insulin', width=500, height=750)
fig.update_layout(xaxis={'categoryorder':'total descending'})
fig.show()


# ## Glyburide-metformin

# In[323]:


print (f'{round(df["glyburide-metformin"].value_counts(normalize=True)*100,2)}')
fig = px.histogram(df, x="glyburide-metformin", title='glyburide-metformin', width=750, height=750)
fig.update_layout(xaxis={'categoryorder':'total descending'})
fig.show()


# ## Glipizide-metformin

# In[324]:


print (f'{round(df["glipizide-metformin"].value_counts(normalize=True)*100,2)}')
fig = px.histogram(df, x="glipizide-metformin", title='glipizide-metformin', width=750, height=750)
fig.update_layout(xaxis={'categoryorder':'total descending'})
fig.show()


# ## Glimepiride-pioglitazone

# In[325]:


print (f'{round(df["glimepiride-pioglitazone"].value_counts(normalize=True)*100,2)}')
fig = px.histogram(df, x="glimepiride-pioglitazone", title='glimepiride-pioglitazone', width=750, height=750)
fig.update_layout(xaxis={'categoryorder':'total descending'})
fig.show()


# ## Metformin-rosiglitazone

# In[326]:


print (f'{round(df["metformin-rosiglitazone"].value_counts(normalize=True)*100,2)}')
fig = px.histogram(df, x="metformin-rosiglitazone", title='metformin-rosiglitazone', width=750, height=750)
fig.update_layout(xaxis={'categoryorder':'total descending'})
fig.show()


# ## Metformin-pioglitazone

# In[327]:


print (f'{round(df["metformin-pioglitazone"].value_counts(normalize=True)*100,2)}')
fig = px.histogram(df, x="metformin-pioglitazone", title='metformin-pioglitazone', width=750, height=750)
fig.update_layout(xaxis={'categoryorder':'total descending'})
fig.show()


# ## Change

# In[329]:


print (f'{round(df["change"].value_counts(normalize=True)*100,2)}')
fig = px.histogram(df, x="change", title='change', width=500, height=750)
fig.update_layout(xaxis={'categoryorder':'total descending'})
fig.show()


# ## DiabetesMed

# In[330]:


print (f'{round(df["diabetesMed"].value_counts(normalize=True)*100,2)}')
fig = px.histogram(df, x="diabetesMed", title='diabetesMed', width=500, height=750)
fig.update_layout(xaxis={'categoryorder':'total descending'})
fig.show()


# ## Readmitted

# In[331]:


print (f'{round(df["readmitted"].value_counts(normalize=True)*100,2)}')
fig = px.histogram(df, x="readmitted", title='readmitted', width=500, height=750)
fig.update_layout(xaxis={'categoryorder':'total descending'})
fig.show()
