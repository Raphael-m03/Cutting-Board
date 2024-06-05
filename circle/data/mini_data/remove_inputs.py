#This file removes the Inputs for there Outputs contains the same data
import csv

def filter_csv(input_file, output_file):
  with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    # Write the header row (assuming there's a header)
    writer.writerow(next(reader))

    for row in reader:
      if not row[0].startswith("Input_"):
        writer.writerow(row)

# Example usage
input_file = "/Users/raphael/Desktop/circle_test/nesting_algorithms/circle/data/mini_data/updated_treated_data.csv"
output_file = "filtered_output.csv"
filter_csv(input_file, output_file)

print(f"CSV filtered, results written to: {output_file}")
