import random
import csv
from typing import List
from sklearn.preprocessing import MinMaxScaler
from rectangleNesting import Rectangle, Frame, Algorithm
import numpy as np

min_dimension = 2
max_dimension = 20
num_rectangles_range = (2, 20)
frame_width = 100
frame_height = 100
num_data_points = 100000

with open("/Users/raphael/Desktop/circle_test/nesting_algorithms/rectangle/data/mini_data/rectangle_nesting_data.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)

    writer.writerow(["test_case", "w", "h", "x", "y", "frame_width", "frame_height", "rotation"])

    for id in range(num_data_points):
        print(id)
        num_rectangles = random.randint(*num_rectangles_range)

        rectangles_list : List[Rectangle] = []

        for i in range(num_rectangles):
            width = random.randint(min_dimension, max_dimension)
            height = random.randint(min_dimension, max_dimension)
            rectangles_list.append(Rectangle(width, height))

        nest = Algorithm(frame_width, frame_height, 0, 0)
        nest.setRectangles(rectangles_list)
        nest.run()
        nested = nest.get_new_rectangle()

        unnested = 20 - len(nested)

        #writer.writerow([f"Input_{id}", None, None, None, frame_width, frame_height, None])
        for n in range(len(nested)):
            x1 = nested[n].x
            y1 = nested[n].y
            w = nested[n].width
            h = nested[n].height
            writer.writerow([f"Output_{id}", nested[n].width, nested[n].height, x1, y1, frame_width, frame_height, nested[n].rotate * 1])

        for j in range(unnested):
            writer.writerow([f"Output_{id}", -1, -1, -1, -1, None, None, -1])


print("Rectangle nesting data generated and saved to rectangle_nesting_data.csv")