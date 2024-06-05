#this file normalize radius
#r_norm : added column where r_norm = R / 20 for R in r and R > 0

import csv
import pandas as pd

def normalize_radii(input_file, output_file):
    data = pd.read_csv(input_file)
    data['r_norm'] = data['radius'].apply(lambda r: r / 20 if r > 0 else r)
    data.to_csv(output_file, index=False)

input_file = "/Users/raphael/Desktop/circle_test/filtered_output.csv"
output_file = "/Users/raphael/Desktop/circle_test/filtered_output.csv"
normalize_radii(input_file, output_file)

print(f"CSV normalized and updated, results written to: {output_file}")