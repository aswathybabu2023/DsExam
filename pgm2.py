import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

dataset=pd.read_csv('slr.csv')
x=dataset.iloc[:,:1].values
y=dataset.iloc[:,2].values

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=42)
reg=LinearRegression()
reg.fit(x_train,y_train)
pred=reg.predict(x_test)

plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,reg.predict(x_train),color='blue')

plt.title("training data ")
plt.xlabel("observations")
plt.ylabel("sales")
plt.show()

plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,reg.predict(x_train),color='blue')

plt.title("test details")
plt.xlabel("observations")
plt.ylabel("sales")
plt.show()

mae=metrics.mean_absolute_error(y_test,pred)
mse=metrics.mean_squared_error(y_test,pred)
rmse=np.sqrt(mse)

print("performance evaluation:\n")
print("MSE: ",mse)
print("MAE :",mae)
print("RMSE :",rmse)