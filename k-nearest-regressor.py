import pandas as pd
import yaml
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor  # Import KNeighborsRegressor
from sklearn import metrics

with open('config.yml', 'r') as f:
    dictConfig = yaml.load(f, Loader=yaml.FullLoader)

customers = pd.read_csv(f"{dictConfig['path']['data']}Ecommerce_Customers.csv", index_col=0)
print(customers.head())

y = customers['Yearly Amount Spent']
X = customers[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# Instantiate the KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)

# Predictions
predictions = knn.predict(X_test)
plt.scatter(y_test, predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.show()

# Evaluate model
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
