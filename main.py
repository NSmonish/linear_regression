import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
#Creating a dataframe
data={"NAME":["Monish","Krish","Jack","Roy","Haaland","Kevin","Hans","Roy","Tapan","Harish"],"GPA":[8.2,9.0,7.15,9.24,8.32,9.36,8.25,8.77,6.53,9.57],"CGPA":[7.91,8.95,7.2,9.35,8.30,9.15,8.76,8.77,6.42,9.38]}
df=pd.DataFrame(data,index=[1,2,3,4,5,6,7,8,9,10])
df.to_csv('linear_regression.csv')
df=pd.read_csv('linear_regression.csv')
df
#fit to linear regression
x=df[["GPA"]]
y=df[["CGPA"]]
regressor=LinearRegression()
regressor.fit(x,y)
print(regressor)
#Visualization
plt.scatter(x,y,color='red')
plt.plot(x,regressor.predict(x),color='blue')
plt.title('GPA vs CGPA')
plt.xlabel('GPA')
plt.ylabel('CGPA')
plt.show()
#Prectiction
r_sq=regressor.score(x,y)
print("coefficient of determination:",r_sq)
intercept=regressor.intercept_
print("Intercept",intercept[0])
slope=regressor.coef_
print("Slope",slope[0][0])
pred=float(input("Enter your GPA: "))
y_pred1=intercept[0]+(slope[0][0])*pred
print("Your predicted CGPA is: ",y_pred1)
