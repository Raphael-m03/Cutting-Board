import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer

data = pd.read_csv('/Users/raphael/Desktop/circle_test/nesting_algorithms/rectangle/data/mini_data/treated_data.csv')

frame_width = 100
frame_height = 100

filtered_data = data[(data['test_case'].str.contains('Output')) & (data['w'] != -1)]

correlation_matrix = filtered_data[['w', 'h', 'loc', 'dist_to_center', 'x_norm', 'y_norm']].corr()
print("Correlation matrix:\n", correlation_matrix)

sns.scatterplot(x='loc', y='w', data=filtered_data)
plt.xlabel('Location (loc)')
plt.ylabel('width')
plt.title('Scatter plot of Radius vs Location')
plt.show()

correlation_matrix = filtered_data[['h', 'loc', 'dist_to_center', 'x_norm', 'y_norm']].corr()
print("Correlation matrix:\n", correlation_matrix)

sns.scatterplot(x='loc', y='h', data=filtered_data)
plt.xlabel('Location (loc)')
plt.ylabel('height')
plt.title('Scatter plot of Radius vs Location')
plt.show()

features = filtered_data[['w', 'h', 'loc', 'dist_to_center', 'x_norm', 'y_norm']]

imputer = SimpleImputer(strategy='mean')
features_imputed = imputer.fit_transform(features)

poly = PolynomialFeatures(degree=2)
poly_features = poly.fit_transform(features_imputed)

print("\nAdded features to df")
poly_df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(['w', 'h', 'loc', 'dist_to_center', 'x_norm', 'y_norm']))
print(poly_df.head())

X = poly_df
y = filtered_data['w']
selector = SelectKBest(score_func=f_regression, k='all')
selector.fit(X, y)
scores = selector.scores_

print("Feature scores:\n", scores)

correlation_matrix = data[['w', 'h', 'x_center', 'y_center', 'loc']].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap between Radius, X, Y, and Loc')
plt.show()

correlation_matrix = data[['w', 'h', 'loc', 'dist_to_center', 'x_norm', 'y_norm', 'x_center', 'y_center', 'rotation']].corr()
print(correlation_matrix)

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Detailed Correlation Heatmap')
plt.show()