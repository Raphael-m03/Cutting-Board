#this file adds the following columns
#x_norm         : normalized x with x_norm = x / 200
#y_norm         : normalized y with y_norm = y / 200
#loc            : loc = y * 200 + x
#dist_to_center : distance from center of nested circle to (0, 0)

import pandas as pd
import numpy as np;

data = pd.read_csv('/Users/raphael/Desktop/circle_test/nesting_algorithms/circle/data/mini_data/untreated_circle_data.csv')
frame_width = 200
frame_height = 200

# Calculate the 'loc' feature
def calculate_loc(row):
    if pd.notna(row['x']) and pd.notna(row['y']) and row['x'] != -1 and row['y'] != -1:
        return row['y'] * frame_width + row['x']
    else:
        return -1

# Apply the function to calculate 'loc' for Output rows
data['loc'] = data.apply(lambda row: calculate_loc(row) if 'Output' in row['test_case'] else pd.NA, axis=1)
    
filtered_data = data[(data['radius'] > 0)].copy()

filtered_data['dist_to_center'] = np.sqrt((filtered_data['x'] - frame_width / 2)**2 + (filtered_data['y'] - frame_height / 2)**2)

# Normalize x and y coordinates
filtered_data['x_norm'] = filtered_data['x'] / frame_width
filtered_data['y_norm'] = filtered_data['y'] / frame_height

data = data.merge(filtered_data[['test_case', 'radius', 'x', 'y', 'dist_to_center', 'x_norm', 'y_norm']], on=['test_case', 'radius', 'x', 'y'], how='left')

data['dist_to_center'].fillna(0, inplace=True)
data['x_norm'].fillna(0, inplace=True)
data['y_norm'].fillna(0, inplace=True)

data.to_csv('/Users/raphael/Desktop/circle_test/nesting_algorithms/circle/data/mini_data/treated_data.csv', index=False)