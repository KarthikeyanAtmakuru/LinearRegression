'''
| Experience | Salary |
| ---------- | ------ |
| 1          | 40000  |
| 2          | 42000  |
| 3          | 45000  |
| 4          | 48000  |
| 5          | 50000  |

Train a Linear Regression model.
Print the slope and intercept.
Predict the salary for 6 years of experience.
Print the R² score.
'''
import pandas as pd
from sklearn.linear_model import LinearRegression
data=pd.DataFrame({
    'Experience':[1,2,3,4,5],
    'Salary':[40000,42000,45000,48000,50000]
})
df=pd.DataFrame(data)
print(df)
# feature and Targets of the model
X=df[['Experience']]
y=df['Salary']
#declaring model
model=LinearRegression()
#fitting the model..
model.fit(X,y)

#print the slope and intercept of model

print('Slope:',model.coef_)
print('Intercept',model.intercept_)

#Predict the salary for 6 years of experience.
new_X=pd.DataFrame({
    'Experience':[6,8]})
predicted_salary=model.predict(new_X)
print('predicted Salary:',predicted_salary)

#Check R2-score
r2=model.score(X,y)
print("Coefficent of Determination",r2)


