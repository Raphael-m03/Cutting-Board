import csv
import circleNesting

def read_csv(filepath, nb_test_cases):
    with open(filepath, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)

        test_cases = []
        current_test_case = []
        current_test_case_number = 0

        for row in reader:
            try:
                test_case_number = int(row[0].split("_")[-1])
            except (IndexError, ValueError):
                continue
            
            if test_case_number >= nb_test_cases + 1:
                break
            
            if(current_test_case_number == 20):
                test_cases.append(current_test_case)
                current_test_case = []
                current_test_case_number = 0

            radius = float(row[1])
            x = row[2]
            y = row[3]
            current_test_case.append([radius, x, y])
            current_test_case_number += 1

    return test_cases

# Example usage
filepath = "/Users/raphael/Desktop/circle_test/filtered_output.csv"
nb_test_cases = 2
test_cases = read_csv(filepath, nb_test_cases)

print(test_cases)