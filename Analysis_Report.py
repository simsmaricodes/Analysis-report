#!/usr/bin/env python
# coding: utf-8

#

# <h1> Library and Data Imports</h1>

# In[1]:


# Importing libraries Essentials
import pandas as pd  # Data science essentials
import matplotlib.pyplot as plt  # Essential graphical output
import seaborn as sns  # Enhanced graphical output
import numpy as np  # Mathematical essentials

# Importing libraries for linear regression
import statsmodels.formula.api as smf  # Regression modeling
from os import listdir  # Look inside file directory
from sklearn.model_selection import train_test_split  # Split data into training and testing data
from sklearn.linear_model import LinearRegression  # OLS Regression
import sklearn.linear_model  # Linear models

# LIbrary for KNN
from sklearn.neighbors import KNeighborsRegressor  # KNN for Regression
from sklearn.preprocessing import StandardScaler  # standard scaler

# Libraries for Logistic Regression
from sklearn.linear_model import LogisticRegression  # logistic regression
import statsmodels.formula.api as smf  # logistic regression
from sklearn.metrics import confusion_matrix  # confusion matrix
from sklearn.metrics import roc_auc_score  # auc score

# libraries for classification trees
from sklearn.tree import DecisionTreeClassifier  # classification trees
from sklearn.tree import export_graphviz  # exports graphics
from six import StringIO  # saves objects in memory
from IPython.display import Image  # displays on frontend
import pydotplus  # interprets dot objects

# Libraries for optimisations
from sklearn.model_selection import RandomizedSearchCV  # hyperparameter tuning
from sklearn.metrics import make_scorer  # customizable scorer
from sklearn.metrics import confusion_matrix  # confusion matrix

# new packages GBM and RAndom Forest
from sklearn.ensemble import RandomForestClassifier  # random forest
from sklearn.ensemble import GradientBoostingClassifier  # gbm

# setting pandas print options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Filepath
file = './Apprentice_Chef_Dataset.xlsx'

# Importing the dataset
apprentice = pd.read_excel(io=file)


# <h1> Function Definitions

# In[2]:


def text_split_feature(col, df, sep=' ', new_col_name='number_of_names'):
    """
Splits values in a string Series (as part of a DataFrame) and sums the number
of resulting items. Automatically appends summed column to original DataFrame.

PARAMETERS
----------
col          : column to split
df           : DataFrame where column is located
sep          : string sequence to split by, default ' '
new_col_name : name of new column after summing split, default
               'number_of_names'
"""

    df[new_col_name] = 0

    for index, val in df.iterrows():
        df.loc[index, new_col_name] = len(df.loc[index, col].split(sep=' '))


# Splitting the names and summing the number of resulting items
text_split_feature(col='NAME',
                   df=apprentice)


# In[3]:


########################################
# visual_cm
########################################
def visual_cm(true_y, pred_y, labels=None):
    """
Creates a visualization of a confusion matrix.

PARAMETERS
----------
true_y : true values for the response variable
pred_y : predicted values for the response variable
labels : , default None
    """
    # visualizing the confusion matrix

    # setting labels
    lbls = labels

    # declaring a confusion matrix object
    cm = confusion_matrix(y_true=true_y,
                          y_pred=pred_y)

    # heatmap
    sns.heatmap(cm,
                annot=True,
                xticklabels=lbls,
                yticklabels=lbls,
                cmap='Blues',
                fmt='g')

    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix of the Classifier')
    plt.show()


# In[4]:


########################################
# display_tree
########################################
def display_tree(tree, feature_df, height=500, width=800):
    """
    PARAMETERS
    ----------
    tree       : fitted tree model object
        fitted CART model to visualized
    feature_df : DataFrame
        DataFrame of explanatory features (used to generate labels)
    height     : int, default 500
        height in pixels to which to constrain image in html
    width      : int, default 800
        width in pixels to which to constrain image in html
    """

    # visualizing the tree
    dot_data = StringIO()

    # exporting tree to graphviz
    export_graphviz(decision_tree=tree,
                    out_file=dot_data,
                    filled=True,
                    rounded=True,
                    special_characters=True,
                    feature_names=feature_df.columns)

    # declaring a graph object
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

    # creating image
    img = Image(graph.create_png(),
                height=height,
                width=width)

    return img


########################################
# plot_feature_importances
########################################
def plot_feature_importances(model, train, export=False):
    """
    Plots the importance of features from a CART model.

    PARAMETERS
    ----------
    model  : CART model
    train  : explanatory variable training data
    export : whether or not to export as a .png image, default False
    """

    # declaring the number
    n_features = x_train.shape[1]

    # setting plot window
    fig, ax = plt.subplots(figsize=(12, 9))

    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(pd.np.arange(n_features), train.columns)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")

    if export == True:
        plt.savefig('Tree_Leaf_50_Feature_Importance.png')


# <h1>Feature Engineering </h1>

# In[5]:


# Log transforms of distorted data

inter_list = ['REVENUE', 'LARGEST_ORDER_SIZE', 'PRODUCT_CATEGORIES_VIEWED', 'PC_LOGINS',
              'TOTAL_MEALS_ORDERED', 'UNIQUE_MEALS_PURCH', 'CONTACTS_W_CUSTOMER_SERVICE',
              'AVG_CLICKS_PER_VISIT']

for item in inter_list:
    # Converting to logs and seeing if the data improves
    apprentice['log_' + item] = np.log10(apprentice[item])

# In[6]:


# STEP 1: splitting personal emails

# placeholder list
placeholder_lst = []

# looping over each email address
for index, col in apprentice.iterrows():
    # splitting email domain at '@'
    split_email = apprentice.loc[index, 'EMAIL'].split(sep='@')

    # appending placeholder_lst with the results
    placeholder_lst.append(split_email)

# converting placeholder_lst into a DataFrame
email_df = pd.DataFrame(placeholder_lst)

# STEP 2: concatenating with original DataFrame
# renaming column to concatenate
email_df.columns = ['0', 'personal_email_domain']

# concatenating personal_email_domain with friends DataFrame
apprentice = pd.concat([apprentice, email_df['personal_email_domain']],
                       axis=1)

# email domain types
personal_email_domains = ['@gmail.com', '@yahoo.com', '@protonmail.com']

# Other Emails
other_email_domains = ['@me.com', '@aol.com', '@live.com', '@passport.com',
                       '@msn.com', '@hotmail.com']

# Domain list
domain_lst = []

# looping to group observations by domain type
for domain in apprentice['personal_email_domain']:
    if '@' + domain in personal_email_domains:
        domain_lst.append('personal')

    elif '@' + domain in other_email_domains:
        domain_lst.append('other')

    else:
        domain_lst.append('work')

# concatenating with original DataFrame
apprentice['domain_group'] = pd.Series(domain_lst)

# checking results
apprentice['domain_group'].value_counts()

# one hot encoding categorical variables
one_hot_domain = pd.get_dummies(apprentice['domain_group'])

# joining codings together
apprentice = apprentice.join([one_hot_domain])

# Game of throme names
got_name = ['Sand', 'Stark', 'Martell', 'Greyjoy',
            'Tully', 'Snow', 'Lannister', 'Baratheon',
            'Frey', 'Tyrell', 'Targaryen', 'Arryn']

apprentice['GOT'] = 0

# LOop
for index, value in apprentice.iterrows():

    # Placing in the new list
    if apprentice.loc[index, 'FAMILY_NAME'] in got_name:
        apprentice.loc[index, 'GOT'] = 1

# log transforming Sale_Price and saving it to the dataset
apprentice['log_REVENUE'] = np.log10(apprentice['REVENUE'])

# Log transforms

inter_list = ['LARGEST_ORDER_SIZE', 'PRODUCT_CATEGORIES_VIEWED', 'PC_LOGINS',
              'TOTAL_MEALS_ORDERED', 'UNIQUE_MEALS_PURCH', 'CONTACTS_W_CUSTOMER_SERVICE']

# Dropping Object Data
apprentice = apprentice.drop(['NAME', 'EMAIL', 'FIRST_NAME',
                              'FAMILY_NAME', 'personal_email_domain',
                              'domain_group'], axis=1)

# In[7]:


# Dummy Variables for the factors we found above with at leasst 100 observations
apprentice['noon_canc'] = 0
apprentice['after_canc'] = 0
apprentice['weekly_plan_sub'] = 0
apprentice['early_delivery'] = 0
apprentice['late_delivery'] = 0
apprentice['masterclass_att'] = 0
apprentice['view_photo'] = 0

# Iter over eachg column to get the new boolean feature columns
for index, value in apprentice.iterrows():

    # For noon cancellations
    if apprentice.loc[index, 'CANCELLATIONS_BEFORE_NOON'] > 0:
        apprentice.loc[index, 'noon_canc'] = 1

    # For afternoon cancelations
    if apprentice.loc[index, 'CANCELLATIONS_AFTER_NOON'] > 0:
        apprentice.loc[index, 'after_canc'] = 1

    # Weekly meal plan subscription
    if apprentice.loc[index, 'WEEKLY_PLAN'] > 0:
        apprentice.loc[index, 'weekly_plan_sub'] = 1

    # Early deliveries
    if apprentice.loc[index, 'EARLY_DELIVERIES'] > 0:
        apprentice.loc[index, 'early_delivery'] = 1

    # Late Deliveries
    if apprentice.loc[index, 'LATE_DELIVERIES'] > 0:
        apprentice.loc[index, 'late_delivery'] = 1

    # Masterclass attendance
    if apprentice.loc[index, 'MASTER_CLASSES_ATTENDED'] > 0:
        apprentice.loc[index, 'masterclass_att'] = 1

    # Viewed Photos
    if apprentice.loc[index, 'TOTAL_PHOTOS_VIEWED'] > 0:
        apprentice.loc[index, 'view_photo'] = 1

# Checking distribution
contact_greater = []
mobile_greater = []

# Instantiating dummy variables
for index, value in apprentice.iterrows():

    # For noon cancellations
    if apprentice.loc[index, 'CONTACTS_W_CUSTOMER_SERVICE'] > (apprentice.loc[index, 'TOTAL_MEALS_ORDERED']) / 2:
        contact_greater.append(1)
    else:
        contact_greater.append(0)

# Instantiating dummy variables
for index, value in apprentice.iterrows():

    if apprentice.loc[index, 'MOBILE_LOGINS'] > apprentice.loc[index, 'PC_LOGINS']:
        mobile_greater.append(1)

    else:
        mobile_greater.append(0)

contact_greater = pd.DataFrame(contact_greater)
mobile_greater = pd.DataFrame(mobile_greater)  # PC logins are consistently more so we dop

contact_greater.value_counts()  # Checking distribution of zeros

# Adding them to the data
apprentice['contact_greater'] = contact_greater
apprentice['mobile_greater'] = mobile_greater

# In[8]:


# Dummy Variables for the factors we found above with at leasst 100 observations
apprentice['meals_below_fif'] = 0
apprentice['meals_above_two'] = 0
apprentice['unique_meals_above_ten'] = 0
apprentice['cust_serv_under_ten'] = 0
apprentice['click_under_eight'] = 0

# Iter over eachg column to get the new boolean feature columns

for index, value in apprentice.iterrows():

    # Total meals greater than 200
    if apprentice.loc[index, 'TOTAL_MEALS_ORDERED'] >= 200:
        apprentice.loc[index, 'meals_below_fif'] = 1

    # Total meals less than 15
    if apprentice.loc[index, 'TOTAL_MEALS_ORDERED'] <= 15:
        apprentice.loc[index, 'meals_above_two'] = 1

    # Unique meals greater 10
    if apprentice.loc[index, 'UNIQUE_MEALS_PURCH'] > 10:
        apprentice.loc[index, 'unique_meals_above_ten'] = 1

    # Customer service less than 10
    if apprentice.loc[index, 'CONTACTS_W_CUSTOMER_SERVICE'] < 10:
        apprentice.loc[index, 'cust_serv_under_ten'] = 1

    # Clicks below 8
    if apprentice.loc[index, 'AVG_CLICKS_PER_VISIT'] < 8:
        apprentice.loc[index, 'click_under_eight'] = 1

# Adding the new variable
apprentice['freq_customer_service'] = 0

# Instantiating dummy variables
for index, value in apprentice.iterrows():

    # For noon cancellations
    if apprentice.loc[index, 'CONTACTS_W_CUSTOMER_SERVICE'] > (apprentice.loc[index, 'TOTAL_MEALS_ORDERED']) / 2:
        apprentice.loc[index, 'freq_customer_service'] = 1

# More features
apprentice['other_many_names'] = 0

for index, value in apprentice.iterrows():

    if apprentice.loc[index, 'other'] == 1 and apprentice.loc[index, 'number_of_names'] == 1:
        apprentice.loc[index, 'other_many_names'] = 1

# In[9]:


# Flags based on Decision tree
# Gini suggestions
apprentice['Before_noon_cancellations'] = 0
apprentice['No_personal_usage'] = 0
apprentice['Low_customer_service_cont'] = 0
apprentice['Low_AVG_click'] = 0
apprentice['Total_meals'] = 0
apprentice['Late_delivery'] = 0
apprentice['Low_revenue'] = 0

# For Loop based on Decision Tree
for index, value in apprentice.iterrows():

    # Noon cancellations <1.5
    if apprentice.loc[index, 'CANCELLATIONS_BEFORE_NOON'] <= 1.5:
        apprentice.loc[index, 'Before_noon_cancellations'] = 1

    # Work below 1.5
    if apprentice.loc[index, 'work'] <= 0.5:
        apprentice.loc[index, 'No_personal_usage'] = 1

    # low total meals
    if apprentice.loc[index, 'TOTAL_MEALS_ORDERED'] <= 47.5:
        apprentice.loc[index, 'Total_meals'] = 1

    # late delivery
    if apprentice.loc[index, 'LATE_DELIVERIES'] <= 1.5:
        apprentice.loc[index, 'Late_delivery'] = 1

    # Low revenue
    if apprentice.loc[index, 'REVENUE'] <= 3987:
        apprentice.loc[index, 'Low_revenue'] = 1

# Creating columns for totals

apprentice['Total_Cancellations'] = apprentice['CANCELLATIONS_BEFORE_NOON'] + apprentice['CANCELLATIONS_AFTER_NOON']

apprentice['Total_Logins'] = apprentice['PC_LOGINS'] + apprentice['MOBILE_LOGINS']

# Creating flags for the new variables
apprentice['out_Total_Cancellations'] = 0
apprentice['out_Total_Logins'] = 0

# Summing the totals
for index, value in apprentice.iterrows():

    # Out cancellations
    if apprentice.loc[index, 'Total_Cancellations'] == 0:
        apprentice.loc[index, 'out_Total_Cancellations'] = 1

        # Out Logins
    if apprentice.loc[index, 'Total_Logins'] >= 7:
        apprentice.loc[index, 'out_Total_Logins'] = 1

    # Average order size in dollars
apprentice['AVG_ORDER_REV'] = apprentice['REVENUE'] / apprentice['TOTAL_MEALS_ORDERED']

# Average money spent on a meal
apprentice['AVG_ORDER_SIZE_REV'] = apprentice['REVENUE'] / apprentice['UNIQUE_MEALS_PURCH']

# Gini suggestions
apprentice['Average_Money_Spent'] = 0
apprentice['Average_Spent_per_meal'] = 0

for index, value in apprentice.iterrows():

    if apprentice.loc[index, 'AVG_ORDER_REV'] <= 110:
        apprentice.loc[index, 'Average_Money_Spent'] = 1

    if apprentice.loc[index, 'AVG_ORDER_SIZE_REV'] >= 4000:
        apprentice.loc[index, 'Average_Spent_per_meal'] = 1

    # <h2> Linear Regression </h2>

# In[10]:


# preparing explanatory variable data

x_variables = ['CROSS_SELL_SUCCESS', 'UNIQUE_MEALS_PURCH', 'CONTACTS_W_CUSTOMER_SERVICE',
               'PRODUCT_CATEGORIES_VIEWED', 'AVG_PREP_VID_TIME', 'LARGEST_ORDER_SIZE',
               'MEDIAN_MEAL_RATING', 'AVG_CLICKS_PER_VISIT', 'masterclass_att',
               'view_photo', 'meals_below_fif', 'log_CONTACTS_W_CUSTOMER_SERVICE', 'log_AVG_CLICKS_PER_VISIT',
               'meals_above_two', 'unique_meals_above_ten', 'click_under_eight',
               'freq_customer_service', 'log_LARGEST_ORDER_SIZE', 'log_PRODUCT_CATEGORIES_VIEWED',
               'log_TOTAL_MEALS_ORDERED', 'log_UNIQUE_MEALS_PURCH', 'log_CONTACTS_W_CUSTOMER_SERVICE',
               'personal', 'work', 'other']

apprentice_data = apprentice[x_variables]

# preparing the target variable
apprentice_target = apprentice.loc[:, 'log_REVENUE']

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(
    apprentice_data,
    apprentice_target,
    test_size=0.25,
    random_state=219)

# INSTANTIATING a model object
lr = LinearRegression()

# FITTING to the training data
lr_fit = lr.fit(X_train, y_train)

# PREDICTING on new data
lr_pred = lr_fit.predict(X_test)

lr_train_score = lr.score(X_train, y_train).round(3)
lr_test_score = lr.score(X_test, y_test).round(3)

print(lr_train_score)

# <h2> Logistic Regression </h2>

# In[11]:


# Removal to include important variables after feature importance
c_var = ['TOTAL_MEALS_ORDERED', 'CANCELLATIONS_BEFORE_NOON', 'AVG_PREP_VID_TIME', 'GOT',
         'No_personal_usage', 'log_REVENUE', 'number_of_names', 'other', 'Before_noon_cancellations',
         'No_personal_usage', 'other_many_names', 'Total_meals', 'masterclass_att']

# train/test split with the logit_sig variables
apprentice_data = apprentice.loc[:, c_var]
apprentice_target = apprentice.loc[:, 'CROSS_SELL_SUCCESS']

# train/test split
x_train, x_test, y_train, y_test = train_test_split(
    apprentice_data,
    apprentice_target,
    random_state=219,
    test_size=0.25,
    stratify=apprentice_target)

# INSTANTIATING a classification tree object
pruned_tree = DecisionTreeClassifier(max_depth=3,
                                     min_samples_leaf=5,
                                     random_state=219)

# FITTING the training data
pruned_tree_fit = pruned_tree.fit(apprentice_data, apprentice_target)

# PREDICTING on new data
pruned_tree_pred = pruned_tree_fit.predict(x_test)

# unpacking the confusion matrix
tuned_tree_tn, tuned_tree_fp, tuned_tree_fn, tuned_tree_tp = confusion_matrix(y_true=y_test,
                                                                              y_pred=pruned_tree_pred).ravel()

# declaring model performance objects
tree_train_acc = pruned_tree.score(x_train, y_train).round(3)
tree_test_acc = pruned_tree.score(x_test, y_test).round(3)
tree_auc = roc_auc_score(y_true=y_test,
                         y_score=pruned_tree_pred).round(3)

# <h1> Intro: Analysis Report to Management </h1>
# <h2> Apprentice Chef Business Case: Management Report</h2>
# <h3> Simbarashe David-Nigel Mariwande <br>

# Apprentice chef is attempting to understand what pushes growth in terms of revenue as well as uptake in promotional content. An analytical approach has been done and resulted in  the following

# <h1>Insights </h1>
# <h2> Revenue</h2>

# In[12]:


########################
# Visual EDA (Scatterplots)
########################

# setting figure size
fig, ax = plt.subplots(figsize=(15, 8))

# developing a scatterplot
plt.subplot(1, 2, 1)
sns.regplot(x=apprentice['AVG_PREP_VID_TIME'],
            y=apprentice['REVENUE'],
            color='g')

# adding labels but not adding title
plt.title(label='Average Prep video vs Revenue')
plt.xlabel(xlabel='Average Video Time')
plt.ylabel(ylabel='Revenue')

########################


# developing a scatterplot
plt.subplot(1, 2, 2)
sns.regplot(x=apprentice['log_TOTAL_MEALS_ORDERED'],
            y=apprentice['REVENUE'],
            color='r')

# adding labels but not adding title
plt.title(label='Total Meals vs Revenue')
plt.xlabel(xlabel='Total Meals')
plt.ylabel(ylabel='Revenue')

# cleaning up the layout and displaying the results
plt.tight_layout()
plt.show()

df_corr = apprentice.corr()['REVENUE']
a = df_corr.loc['AVG_PREP_VID_TIME'].round(decimals=2)
b = df_corr.loc['log_TOTAL_MEALS_ORDERED'].round(decimals=2)

print(f"""

The Following Correlations are the highest in the Linear regression model:

Average Video prep time     : {a}
Total NUmber of Orders Made : {b}

""")

# From the regression model and analysis it can be gleaned that the average video time and total meals ordered have a positive effect on the revenue brought into apprentice chef. These variables also have high correlations with Revenue and as can be seen on in the above figures.
# <br>
# <h2>Cross-Sell Promotion </h2>

# In[13]:


# Feature importance
x_important = ["Other Email", "Cancellations before noon"]
y_important = [0.41, 0.26]

# setting figure size
fig, ax = plt.subplots(figsize=(15, 8))

# Plotting
sns.barplot(x=x_important,
            y=y_important)

# adding labels but not adding title
plt.title(label='Feature Importance')
plt.xlabel(xlabel='Features')
plt.ylabel(ylabel='Importance')

# Plotting
plt.show()

print("""
Other Email Domains :
Me.com
Aol.com
Live.com
Passport.com
Passport.com
Hotmail
""")

# Other email tag is defined above.The feature importance graph shows that users with these email addresses were most receptive to the promotion, this could mean it could be beneficial to market to their email addresses. The cancellation column represents customer engagement. Customers willing to follow cancellation policy correctly are more engaged and are more willing to participate in promos.

# <h1> Model Performances </h1>

# In[15]:


# A print statement containing the model performances
print(f"""

Linear Regression Model R squared   : {lr_train_score.round(3)}
Logistic Regression Model AUC score : {tree_auc.round(3)}

""")

# <h1> Conclusions </h1>

# The next steps for the company should be to prioritise increasing the number of customers who accept the "halfway there" promotion. The promotion is positively related to the  increase in revenue.
# <br><br>
# To increase the promotion uptake, advertising can be done to customers who use emails that are in the other email group as explained in section 5.2. It can also help to develop another algorithm to guauge customer engagement as this also contributes to the promotions success.
# <br><br>
# Finally, polling why customers watch prep videos could help in future content development. It is evident that the more customers watch the videos the more they are inclined to spend.

# <h1> References </h1>
# <br><br>
# Loaiza, S. (2020, March 23). Gini Impurity Measure. Retrieved January 28, 2021, from https://towardsdatascience.com/gini-impurity-measure-dbd3878ead33
# <br>
# Sturtz, J. (2021, January 23). Python "for" Loops (Definite Iteration). Retrieved January 28, 2021, from https://realpython.com/python-for-loop/
# <br>
# S. (n.d.). Statistical data visualization¶. Retrieved January 28, 2021, from https://seaborn.pydata.org/
# <br
#

