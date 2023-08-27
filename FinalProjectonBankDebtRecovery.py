#!/usr/bin/env python
# coding: utf-8

# In[23]:


# Import modules
import pandas as pd
import numpy as np
# Read in dataset
BankData = pd.read_csv('/Applications/bankData1.csv')
# Print the first few rows of the DataFrame
BankData.head()
BankData.tail()


# In[24]:


# Import stats module
from scipy import stats

# Compute average age just below and above the threshold
era_900_1100 = BankData.loc[(BankData['expected_recovery_amount']<1100) &
                      (BankData['expected_recovery_amount']>=900)]
by_recovery_strategy = era_900_1100.groupby(['recovery_strategy'])
by_recovery_strategy['age'].describe().unstack()

# Perform Kruskal-Wallis test
Level_0_age = era_900_1100.loc[BankData['recovery_strategy']=="Level 0 Recovery"]['age']
Level_1_age = era_900_1100.loc[BankData['recovery_strategy']=="Level 1 Recovery"]['age']
stats.kruskal(Level_0_age, Level_1_age)


# In[25]:


# Number of customers in each category
crosstab = pd.crosstab(BankData.loc[(BankData['expected_recovery_amount']<2000) &
                              (BankData['expected_recovery_amount']>=0)]['recovery_strategy'],
                       BankData['sex'])
print(crosstab)

# Chi-square test
chi2_stat, p_val, dof, ex = stats.chi2_contingency(crosstab)
print(p_val)


# In[27]:


# Scatter plot of Actual Recovery Amount vs. Expected Recovery Amount
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.scatter(x=BankData['expected_recovery_amount'], y=BankData['actual_recovery_amount'], c="g", s=2)
plt.xlim(900, 1100)
plt.ylim(0, 2000)
plt.xlabel("Expected_Recovery_Amount")
plt.ylabel("Actual_Recovery_Amount")
plt.legend(loc=2)
plt.show()


# In[33]:


# Perform Kruskal-Wallis test
Level_0_actual = era_900_1100.loc[BankData['recovery_strategy']=='Level 0 Recovery']['actual_recovery_amount']
Level_1_actual = era_900_1100.loc[BankData['recovery_strategy']=='Level 1 Recovery']['actual_recovery_amount']
print(stats.kruskal(Level_0_actual, Level_1_actual))

# Repeat for a smaller range of $950 to $1050
era_950_1050 = BankData.loc[(BankData['expected_recovery_amount']<1050) &
                      (BankData['expected_recovery_amount']>=950)]
Level_0_actual = era_950_1050.loc[BankData['recovery_strategy']=='Level 0 Recovery']['actual_recovery_amount']
Level_1_actual = era_950_1050.loc[BankData['recovery_strategy']=='Level 1 Recovery']['actual_recovery_amount']
stats.kruskal(Level_0_actual, Level_1_actual)


# In[29]:


# Import statsmodels
import statsmodels.api as sm

# Define X and y
X = era_900_1100['expected_recovery_amount']
y = era_900_1100['actual_recovery_amount']
X = sm.add_constant(X)

# Build linear regression model
model = sm.OLS(y, X).fit()
predictions = model.predict(X)

# Print out the model summary statistics
model.summary()


# In[30]:


# Create indicator (0 or 1) for expected recovery amount >= $1000
BankData['indicator_1000'] = np.where(BankData['expected_recovery_amount']<1000, 0, 1)
era_900_1100 = BankData.loc[(BankData['expected_recovery_amount']<1100) &
                      (BankData['expected_recovery_amount']>=900)]

# Define X and y
X = era_900_1100['expected_recovery_amount']
y = era_900_1100['actual_recovery_amount']
X = sm.add_constant(X)

# Build linear regression model
model = sm.OLS(y,X).fit()

# Print the model summary
model.summary()


# In[31]:


# Redefine era_950_1050 so the indicator variable is included
era_950_1050 = BankData.loc[(BankData['expected_recovery_amount']<1050) & (BankData['expected_recovery_amount']>=950)]
# Define X and y
X = era_950_1050[['expected_recovery_amount','indicator_1000']]
y = era_950_1050['actual_recovery_amount']
X = sm.add_constant(X)

# Build linear regression model
model = sm.OLS(y,X).fit()

# Print the model summary
model.summary()

