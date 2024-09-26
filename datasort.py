import pandas as pd

# Display the dataset
real_estate_data = pd.read_csv('real_estate.csv')

# Display the first few rows of dataset and the info about the data
real_estate_data_head = real_estate_data.head()
real_estate_data_info = real_estate_data.info()

# print(real_estate_data_head)
# print(real_estate_data_info)

# Check for any null values in dataset
# print(real_estate_data.isnull().sum())

# Descriptive statistics of the dataset
descriptive_stats = real_estate_data.describe()

# print(descriptive_stats)

# Histograms of all numerical features

import matplotlib.pyplot as plt
import seaborn as sns

# Set the aesthetic style of the plots
sns.set_style("whitegrid")

# Create histograms for the numerical columns
# fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10,10))
# fig.suptitle('Histograms of Real Estate Data', fontsize=16)

# cols = ['House age',
#         'Distance to the nearest MRT station',
#         'Number of convenience stores',
#         'Latitude',
#         'Longitude',
#         'House price of unit area']

# for i,col in enumerate(cols):
#     sns.histplot(real_estate_data[col],kde=True, ax=axes[i//2,i%2])
#     axes[i//2,i%2].set_title(col)
#     axes[i//2,i%2].set_xlabel('')
#     axes[i//2,i%2].set_ylabel('')

# plt.tight_layout(rect=[0,0,0.95,0.95])
# plt.show()
# plt.savefig('Histograms of Real Estate Data')

# Scatter plots to observe the realtionship with house price
# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12,10))
# fig.suptitle('Scatter Plots with House Price of Unit Area', fontsize=16)

# # Scatter plot for each variable against the house price
# sns.scatterplot(data=real_estate_data, x='House age'                            , y='House price of unit area', ax=axes[0,0])
# sns.scatterplot(data=real_estate_data, x='Distance to the nearest MRT station'  , y='House price of unit area', ax=axes[0,1])
# sns.scatterplot(data=real_estate_data, x='Number of convenience stores'         , y='House price of unit area', ax=axes[1,0])
# sns.scatterplot(data=real_estate_data, x='Latitude'                             , y='House price of unit area', ax=axes[1,1])

# plt.tight_layout(rect=[0,0,0.95,0.95])
# plt.show()
# plt.savefig('Scatter Plots with House Price of Unit Area')

# Correlation matrix
# correlation_matrix = real_estate_data.corr()

# Plotting the correlation matrix
# plt.figure(figsize=(10,6))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
# plt.title('Correlation Matrix')
# plt.show()
# plt.savefig('Correlation Matrix')

# print(correlation_matrix)

# Build a regression model to predict the real estate prices by using Linear Regression algorithm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

# Selecting features and target variable
features = ['Distance to the nearest MRT station', 'Number of convenience stores', 'Latitude', 'Longitude']
target = 'House price of unit area'

x = real_estate_data[features]
y = real_estate_data[target]

# Splitting the dataset into training and testing sets
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Model initialization
model = LinearRegression()

# Training the model
model.fit(x_train,y_train)

# making predicitions using the linear regression model
y_pred_lr = model.predict(x_test)

# Visualization: Actual vs. Predicted Values
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred_lr, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(),y_test.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs. Predicted House Prices')
# plt.show()
plt.savefig('Actual vs. Predicted House Prices.png')