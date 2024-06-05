import random
import csv
from typing import List
from sklearn.preprocessing import MinMaxScaler
from circleNesting import Circle, Frame, Algorithm
import numpy as np

min_radius = 2
max_radius = 20
num_circles_range = (2, 20)
frame_width = 200
frame_height = 200
num_data_points = 100000

with open("/Users/raphael/Desktop/circle_test/nesting_algorithms/circle/data/mini_data/untreated_data.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)

    writer.writerow(["test_case", "radius", "x", "y", "frame_width", "frame_height"])

    for id in range(num_data_points):
        print(id)
        num_circles = random.randint(*num_circles_range)

        circle_list : List[Circle] = []

        for i in range(num_circles):
            radius = random.randint(min_radius, max_radius)
            circle_list.append(Circle(radius))

        frame = Frame(frame_width, frame_height)
        nest, not_nested = Algorithm.nest_circles(circle_list, frame)

        unnested = 20 - len(nest)

        #writer.writerow([f"Input_{id}", None, None, None, frame_width, frame_height, None])
        for n in range(len(nest)):
            writer.writerow([f"Input_{id}", nest[n].r, None, None, frame_width, frame_height])

        for j in range(unnested):
            writer.writerow([f"Input_{id}", -1, None, None, None, None])

        for n in range(len(nest)):
            x1 = nest[n].x
            y1 = nest[n].y
            x2 = nest[n-1].x
            y2 = nest[n-1].y
            writer.writerow([f"Output_{id}", nest[n].r, x1, y2, frame_width, frame_height])

        for j in range(unnested):
            writer.writerow([f"Output_{id}", -1, -1, -1, None, None])


print("Circle nesting data generated and saved to circle_nesting_data.csv")