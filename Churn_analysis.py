import numpy as np 
import pandas as pd
import matplotlib.pyplot  as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

#-----------------------------------------------------------------------------------------------------------
# data reading
#-----------------------------------------------------------------------------------------------------------
df = pd.read_csv('C:/Users/vishwa/DSDA1/csv/customer_churn_dataset-testing-master.csv')
print(df.head())

#-----------------------------------------------------------------------------------------------------------
# data cleaning
#-----------------------------------------------------------------------------------------------------------
print(df.isnull().sum())
print(df.describe())
print(df.info())
print(df.nunique())

#-----------------------------------------------------------------------------------------------------------
# data visualization
#-----------------------------------------------------------------------------------------------------------

churn_by_gender = df.groupby('Gender')['Churn'].sum().plot(kind='bar')
plt.show()

churn_by_age =df.groupby('Age')['Churn'].value_counts().plot(kind='line',color='Green')
plt.grid()
plt.xlabel('Age')
plt.ylabel('Counts churn')
plt.legend()
plt.show()

# Pairplot to know relation
sns.pairplot(df,vars=['Age','Tenure','Usage Frequency','Total Spend'],hue='Gender')
plt.show()

# churn by age category of subscriber type
sns.scatterplot(x='Churn',y='Age',hue='Subscription Type',data=df)
plt.show()

# multi chart relation for support calls,payment delay, on basis of Subscripation
sns.pairplot(df,vars=['Age','Support Calls','Payment Delay','Last Interaction'],hue='Subscription Type')
plt.show()

usage_by_age =df.groupby('Contract Length')['Subscription Type'].value_counts()

total_amt_spend = df.groupby('Subscription Type')['Total Spend'].sum()
print(total_amt_spend)

delay_pay = df.groupby('Subscription Type')['Payment Delay'].median()
print(delay_pay)

#----------------------------------------------------------------------------------------------
# linear_reg model to future churn
#----------------------------------------------------------------------------------------------

X = df[['Age']]
y = df['Churn']

model = LinearRegression()
model.fit(X,y)

print('slope::',model.coef_[0])
print('intercept::',model.intercept_)

Age = 55
predicted_Churn = model.predict([[Age]])      
print(f'Predicted churn by Age {Age} Age: {predicted_Churn[0]}')

plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X, model.predict(X), color='red', label='Regression line')
plt.xlabel('Age')
plt.ylabel('Churn')
plt.title('churn by age')
plt.show()

#-------------------------------------------------------------------------------------------------------
# data saving
#-------------------------------------------------------------------------------------------------------

df.to_csv('Churn_Cust_data.csv',index=False)