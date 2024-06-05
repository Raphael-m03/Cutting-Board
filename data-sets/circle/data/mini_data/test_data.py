import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer

data = pd.read_csv('/Users/raphael/Desktop/circle_test/nesting_algorithms/circle/data/mini_data/updated_treated_data.csv')

frame_width = 200
frame_height = 200

filtered_data = data[(data['test_case'].str.contains('Output')) & (data['radius'] != -1)]

filtered_data['dist_to_center'] = np.sqrt((filtered_data['x'])**2 + (filtered_data['y'])**2)

filtered_data['x_norm'] = filtered_data['x'] / frame_width
filtered_data['y_norm'] = filtered_data['y'] / frame_height
filtered_data['r_norm'] = filtered_data['radius'] / 20

correlation = filtered_data['radius'].corr(filtered_data['loc'])
print("Correlation between radius and loc:", correlation)
correlation = filtered_data['radius'].corr(filtered_data['x'])
print("Correlation between radius and x:", correlation)
correlation = filtered_data['radius'].corr(filtered_data['y'])
print("Correlation between radius and y:", correlation)
correlation = filtered_data['r_norm'].corr(filtered_data['x_norm'])
print("Correlation between radius norm and x norm:", correlation)
correlation = filtered_data['r_norm'].corr(filtered_data['y_norm'])
print("Correlation between radius norm and y norm:", correlation)

correlation_matrix = filtered_data[['radius', 'loc', 'dist_to_center', 'x_norm', 'y_norm']].corr()
print("Correlation matrix:\n", correlation_matrix)

sns.scatterplot(x='loc', y='radius', data=filtered_data)
plt.xlabel('Location (loc)')
plt.ylabel('Radius')
plt.title('Scatter plot of Radius vs Location')
plt.show()

features = filtered_data[['radius', 'loc', 'dist_to_center', 'x_norm', 'y_norm']]

imputer = SimpleImputer(strategy='mean')
features_imputed = imputer.fit_transform(features)

poly = PolynomialFeatures(degree=2)
poly_features = poly.fit_transform(features_imputed)

print("\nAdded features to df")
poly_df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(['radius', 'loc', 'dist_to_center', 'x_norm', 'y_norm']))
print(poly_df.head())

X = poly_df
y = filtered_data['radius']
selector = SelectKBest(score_func=f_regression, k='all')
selector.fit(X, y)
scores = selector.scores_

print("Feature scores:\n", scores)

correlation_matrix = data[['radius', 'x', 'y', 'loc']].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap between Radius, X, Y, and Loc')
plt.show()

correlation_matrix = data[['radius', 'loc', 'dist_to_center', 'x_norm', 'y_norm', 'x', 'y']].corr()
print(correlation_matrix)

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Detailed Correlation Heatmap')
plt.show()