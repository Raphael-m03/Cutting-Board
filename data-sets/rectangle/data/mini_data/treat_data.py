#this file adds the following columns
#x_norm         : normalized x with x_norm = x / 200
#y_norm         : normalized y with y_norm = y / 200
#loc            : loc = y * 100 + x
#dist_to_center : distance from center of nested rectangle to (0, 0)

import pandas as pd
import numpy as np;

data = pd.read_csv('nesting_algorithms/rectangle/data/mini_data/rectangle_nesting_data.csv')
frame_width = 100
frame_height = 100

# Calculate the 'loc' feature
def calculate_loc(row):
    if pd.notna(row['x']) and pd.notna(row['y']) and row['x'] != -1 and row['y'] != -1:
        return (row['y'] + row['h'] / 2) * frame_width + (row['x'] + row['w'] / 2)
    else:
        return -1

# Apply the function to calculate 'loc' for Output rows
data['loc'] = data.apply(lambda row: calculate_loc(row) if 'Output' in row['test_case'] else pd.NA, axis=1)
    
filtered_data = data[(data['w'] > 0)].copy()

filtered_data['dist_to_center'] = np.sqrt((filtered_data['x'])**2 + (filtered_data['y'])**2)

# Normalize x and y coordinates
filtered_data['x_center'] = (filtered_data['x'] + filtered_data['w'] / 2)
filtered_data['y_center'] = (filtered_data['y'] + filtered_data['h'] / 2)
filtered_data['x_norm'] = (filtered_data['x'] + filtered_data['w'] / 2) / frame_width
filtered_data['y_norm'] = (filtered_data['y'] + filtered_data['h'] / 2) / frame_width
filtered_data['w_norm'] = filtered_data['w'] / 20
filtered_data['h_norm'] = filtered_data['h'] / 20

data = data.merge(filtered_data[['test_case', 'w', 'h', 'x', 'y', 'dist_to_center', 'x_norm', 'x_center', 'y_center','y_norm', 'w_norm', 'h_norm']], on=['test_case', 'w', 'h', 'x', 'y'], how='left')

data['dist_to_center'].fillna(0, inplace=True)
data['x_norm'].fillna(0, inplace=True)
data['y_norm'].fillna(0, inplace=True)
data['w_norm'].fillna(0, inplace=True)
data['h_norm'].fillna(0, inplace=True)

data.to_csv('/Users/raphael/Desktop/circle_test/nesting_algorithms/rectangle/data/mini_data/treated_data.csv', index=False)