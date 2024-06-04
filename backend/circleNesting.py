import matplotlib.pyplot as plt
from matplotlib.patches import Circle as Circles
import math
import random
from typing import List, Tuple
import matplotlib.axes._axes as Axes

class Circle:
    def __init__(self, r : float,  thickness : float = 0, x : float = - 1, y : float = -1) -> None:
        self.r: float = 0

        try:
            if isinstance(r, float) or isinstance(r, int):
                if r <= 0:
                    raise ValueError("Radius must be a positive float/int")
                self.r: float = r
        except ValueError as e:
            print(e)

        self.x : float = x
        self.y : float = y
        self.thickness : float = 0

        try:
            if isinstance(thickness, float) or isinstance(thickness, int):
                if thickness < 0 or thickness >= r:
                    raise ValueError("Thickness must be a non-negative float/int less than the radius")
                self.thickness : float = thickness
        except ValueError as e:
            print(e)

        self.__area : float = math.pi * (self.r ** 2)
        self.__empty_area : float = 0

        if(self.thickness > 0):
            self.__nested_circles : List[Circle] = []
            self.__empty_area = math.pi * ((self.r - self.thickness) ** 2)
    
    def add_nested_circle(self, nested_circle : 'Circle') -> bool:
        if(self.thickness > 0 and self.can_add(nested_circle)):
            self.__nested_circles.append(nested_circle)
            return True
        else:
            print("Cannot nest circles in a filled circle or in a smaller empty circle")
            return False

    def get_circle_data(self) -> str:
        string1 = "radius: " + str(self.r) + " thisckness: " + str(self.thickness) + " x: " + str(self.x) + " y: " + str(self.y) + " area: " + str(self.__area)
        string2 = ""
        string3 = ""
        if(self.thickness > 0):
            string2 = "\n\tnumber of nested circles: " + str(len(self.__nested_circles))
            for circle in self.__nested_circles:
                string3 += "\n\t\t" + circle.get_circle_data()
        
        return string1 + string2 + string3
    
    def get_nested_circles(self) -> List['Circle'] : return self.__nested_circles

    def plot_circle(self, ax : Axes) -> None:
        radius = self.r
        thickness = self.thickness
        x = self.x
        y = self.y
        center = (x, y)
    
        color = (random.random(), random.random(), random.random())

        if self.thickness:
            # Draw the colored circle representing the main body
            colored_circle_patch = Circles((x, y), radius, edgecolor='black', facecolor=color)
            ax.add_patch(colored_circle_patch)

            # Draw the white circle on top to simulate thickness
            white_circle_patch = Circles((x, y), radius - self.thickness, edgecolor='black', facecolor='white')
            ax.add_patch(white_circle_patch)
        else:
            # Draw the circle without thickness
            circle_patch = Circles((x, y), radius, edgecolor='black', facecolor=color)
            ax.add_patch(circle_patch)

        # Add a text annotation to mark the center, radius, thickness, and position of the circle
        ax.text(x, y, f"Center: ({x}, {y})\nRadius: {radius}\nThickness: {thickness}\nPosition: ({x}, {y})", ha='center', va='center', fontsize = 5)

        if thickness > 0:
            ax.text(x, y - 2*radius, f"Thickness: {thickness}", ha='center', va='center', fontsize=5)

    def compare_to(self ,circle : 'Circle') -> bool:
        return self.__area > circle.__area

    def can_add(self, circle : 'Circle') -> bool:
        if(self.thickness > 0):
            if isinstance(circle, Circle) and (self.r - self.thickness > circle.r):
                return True
            
        return False

    def get_area(self) -> float:
        return self.__area
    
    def get_empty_area(self) -> float:
        return self.__empty_area

class Frame:
    def __init__(self, width : float, height : float) -> None:
        self.width : float = 0
        self.height : float = 0

        try:
            if isinstance(width, float) or isinstance(width, int) or isinstance(height, float) or isinstance(height, int):
                if height <= 0 or width <= 0:
                    raise ValueError("Dimensions must be a positive float/int")
                self.width : float = width
                self.height : float = height
        except ValueError as e:
            print(e)

        self.__area : float = width * height
        self.__occupied_area : float = 0
        self.__nested_circles : float= []

    def add_circle(self, circle : Circle) -> bool:
        if isinstance(circle, Circle) and (self.__occupied_area + circle.get_area() - circle.get_empty_area() <= self.__area):
            self.__nested_circles.append(circle)
            self.__occupied_area += circle.get_area() - circle.get_empty_area()
            return True
        return False

    def get_area(self) -> float: return self.__area
    
    def get_occupied_area(self) -> float: return self.__occupied_area
    
    def get_nested_circles(self) -> List[Circle]: return self.__nested_circles
    
    def get_frame_data(self) -> str:
        string1 = "width: " + str(self.width) + " height: " + str(self.height) + " area: " + str(self.__area) + " occupied area: " + str(self.__occupied_area)
        string2 = "\n\tnumber of nested circles: " + str(len(self.get_nested_circles()))
        string3 = ""
        for circle in self.get_nested_circles():
            string3 += "\n\t\t" + circle.get_circle_data().replace("\t", "\t\t\t")

        return string1 + string2 + string3

    def plot(self) -> Axes:
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)

        for circle in self.get_nested_circles():
            circle.plot_circle(ax)

        ax.set_title('Nested Circles in Frame')
        ax.set_xlabel('Width')
        ax.set_ylabel('Height')
        ax.grid(True)
        
        plt.show()
        return ax

class Algorithm:
    @staticmethod
    def sort_circles(circles : List[Circle]) -> List[Circle]:
        if all(isinstance(circle, Circle) for circle in circles):
            return sorted(circles, key=lambda circle: (-circle.r, -circle.get_empty_area()))
        else:
            raise ValueError("All elements in the circles list must be instances of the Circle class.")

    @staticmethod
    def sort_circle_by_radius(circles : List[Circle], reverse : bool = False) -> List[Circle]:
        if all(isinstance(circle, Circle) for circle in circles):
            return sorted(circles, key=lambda circle: circle.r, reverse=reverse)
        else:
            raise ValueError("All elements in the circles list must be instances of the Circle class.")
        
    @staticmethod
    def sort_circle_by_empty_area(circles : List[Circle], reverse : bool = False) -> List[Circle]:
        if all(isinstance(circle, Circle) for circle in circles):
            return sorted(circles, key=lambda circle: circle.get_empty_area(), reverse=reverse)
        else:
            raise ValueError("All elements in the circles list must be instances of the Circle class.")

    @staticmethod
    def distance(x1 : float, y1 : float, x2 : float, y2 : float) -> float:
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    @staticmethod
    def calculate_average_radius(circles : List[Circle]) -> float:
        total_radius = 0
        num_circles = len(circles)

        # Sum up the radii of all circles
        for circle in circles:
            total_radius += circle.r

        # Calculate the average radius
        if num_circles > 0:
            average_radius = total_radius / num_circles
            return average_radius
        else:
            return 0 

    @staticmethod
    def nest_circles(circles : List[Circle], frame : Frame, step : float = 1, automatic : bool = False) -> Tuple[List[Circle], List[Circle]]:
        sorted_circles = Algorithm.sort_circles(circles)
        nested = []
        not_nested = []
        inside = []

        for circle in sorted_circles:
            nestable = True

            if((circle.r*2)**2 > frame.get_area()):
                not_nested.append(circle)
                continue

            if(circle.get_area() > frame.get_area() - frame.get_occupied_area()):
                not_nested.append(circle)
                continue

            avg = Algorithm.calculate_average_radius(circles)

            if(circle.r < avg/2 and step == 1 and automatic):
                new_step = step * (circle.r / avg)
                print(new_step)
            else:
                new_step = step

            if(inside):
                nestable = True
                for insider in inside:
                    nestable = True
                    if insider.r - insider.thickness >= circle.r - circle.thickness:
                        minX = insider.x - insider.r + insider.thickness + circle.r
                        maxX = insider.x + insider.r - insider.thickness - circle.r
                        minY = insider.y - insider.r + insider.thickness + circle.r
                        maxY = insider.y + insider.r - insider.thickness - circle.r
                    
                        y = minY - new_step
                        while(y < maxY):
                            y += new_step
                        
                            x = minX - new_step
                            while(x < maxX):
                                x += new_step
                                if(Algorithm.distance(x, y, insider.x, insider.y) > (insider.r - insider.thickness - circle.r)):
                                    nestable = False
                                    continue
                                else:
                                    nestable = True

                                if(insider.get_nested_circles()):
                                    for c in insider.get_nested_circles():
                                        x1 = c.x
                                        y1 = c.y
                                        r1 = c.r
                                        dist = Algorithm.distance(x, y, x1, y1)

                                        if(dist < r1 + circle.r):
                                            nestable = False;
                                            break
                                else:
                                    circle.x = x
                                    circle.y = y
                                    nested.append(circle)
                                    frame.add_circle(circle)
                                    insider.add_nested_circle(circle)
                                    if(circle.thickness > 0):
                                        inside.append(circle)
                                    nestable = True
                                    break

                                if nestable:
                                    circle.x = x
                                    circle.y = y
                                    nested.append(circle)
                                    frame.add_circle(circle)
                                    insider.add_nested_circle(circle)
                                    if(circle.thickness > 0):
                                        inside.append(circle)
                                    nestable = True
                                    break
                                    
                            if nestable:
                                break
                    
                        if nestable:
                            break
                    
                    else:
                        nestable = False
            
            if (not nestable or not inside):
                nestable = True

                y = circle.r - new_step
                while(y < frame.height - circle.r):
                    y += new_step

                    x = circle.r - new_step
                    while(x < frame.width - circle.r):
                        x += new_step
                        if(not nested):
                            if(x + circle.r <= frame.width and y + circle.r <= frame.height):
                                circle.x = circle.r
                                circle.y = circle.r
                                nested.append(circle)
                                frame.add_circle(circle)
                                if(circle.thickness > 0):
                                    inside.append(circle)
                                break
                        else:
                            nestable = True
                            for nest in nested:
                                x1 = nest.x
                                y1 = nest.y
                                r1 = nest.r
                                dist = Algorithm.distance(x, y, x1, y1)
                                if(dist < circle.r + r1  or circle.r + x > frame.width or circle.r + y > frame.height):
                                    nestable = False
                                    break
                        
                            if nestable:
                                circle.x = x
                                circle.y = y
                                nested.append(circle)
                                frame.add_circle(circle)
                                if(circle.thickness > 0):
                                    inside.append(circle)
                                break

                    if(nestable):
                        break

            if(not nestable):
                not_nested.append(circle)

        return nested, not_nested

class Help:
    class circle_class_help:
        @staticmethod
        def circle_definition() -> str:
            string1 = "\nCircle class Documentation:\n\t-Utility: Represents a circle with a given radius and optional thickness."
            string2 = "\n\t-Fields:\n\t\t*Public Fields:\n\t\t\t#r: Radius of the circle, always positive or else value error will be raised.\n\t\t\t#thickness: (Optional) it represent a homogeneous thickness on the membrane of a circle, rendering it similar to a frame with a circular empty area in the middle with a radius of r-thickness, thickness is always positive and <= then r, or else value error will be raised"
            string3 = "\n\t\t*Private Fields:\n\t\t\t#x & y: Represent the position of a circle on the orthogonal axis.\n\t\t\t#area: The area occupied by a circle.\n\t\t\t#empty_area: The hallow area available in a circle with a specified thickness.\n\t\t\t#nested_circles: In a hallow circle with a thickness T, smaller circles can be nested inside it to make the nesting process more efficient."
            string4 = "\n\t-Methods:\n\t\t#add_nested_circle(self, nested_circle : 'Circle') -> bool: Takes a circle and nest it in a greater empty circle, returns false if not possible.\n\t\t#get_circle_data(self) -> str: Returns the circles field values in a string.\n\t\t#plot_circle(self, ax : Axes) -> None: Add the circle to the axe of a matplotlib orthogonal axe.\n\t\t#compare_to(self ,circle : 'Circle') -> bool: compare a another circle with this circle according to area, returns False if the other circle has greater occupied area.\n\t\t#can_add(self, circle : 'Circle') -> bool: Checks if a circle can fit in the hallow circle.\n\t\t#get_area(self) -> float: Returns area.\n\t\t#get_empty_area(self) -> float: Returns unoccupied area\n"

            return string1 + string2 + string3 + string4
        
        @staticmethod
        def usage() -> str:
            string1 = "\nTo create a single circle:\n\tc = Circle(29) #using int or float for radius.\n\tc = Circle(29.25, 1) #using int or float for thickness."
            string2 ="\nTo create multiple circles in a list:\n\tcircles = [\n\t\tCircle(29.25),\n\t\tCircle(29.25, 1),\n\t\tCircle(20),\n\t\tCircle(15)\n\t]\n"

            return string1 + string2

    class frame_class_help:
        @staticmethod
        def frame_definition() -> str:
            string1 = "\nFrame class Documentation:\n\t-Utility: Represents a frame with a given width and height to nest the circles in."
            string2 = "\n\t-Fields:\n\t\t*Public Fields:\n\t\t\t#width & height: Represent the width and height of the frame, always positive or else value error will be raised."
            string3 = "\n\t\t*Private Fields:\n\t\t\t#area: Represent the total area of the frame.\n\t\t\t#occupied_area: Represent the total occupied area by the nested circles.\n\t\t\t#nested_circles: Represent a list of nested circles."
            string4 = "\n\t-Methods:\n\t\t#get_area(self) -> float: Returns area.\n\t\t#get_occupied_area(self) -> float: Returns occupied area by the circles.\n\t\t#get_nested_circles(self) -> List[Circle]: Returns the list of nested circles.\n\t\t#get_frame_data(self) -> str: Returns all data related to frame and nested circles.\n\t\t#plot(self) -> None: Plots the frame and all the nested circles within it.\n"

            return string1 + string2 + string3
            
        @staticmethod
        def usage() -> str:
            return "\nTo create a frame:\n\tframe = Frame(100, 100) #use in or float for width and height.\n"

    class algorithm_class_help:
        @staticmethod
        def algorithm_definition() -> str:
            return "\nAlgorithm class Document:\n\t-Utility: This class contains multiple functions that helps the main function nest_circles() to nest circles using grid laying technique.\n\t-Methods:\n\t\t#sort_circles(circles : List[Circle]) -> List[Circle]: Takes a list of circles and nest them from greater radius to smaller radius, then nests circles with the same radius from greater empty area to smaller empty area, Returns the sorted circles.\n\t\t#sort_circle_by_radius(circles : List[Circle], reverse : bool = False) -> List[Circle]: sort circles by radius.\n\t\t#sort_circle_by_empty_area(circles : List[Circle], reverse : bool = False) -> List[Circle]: sort circles by empty area.\n\t\t#distance(x1 : float, y1 : float, x2 : float, y2 : float) -> float: Returns the distance between 2 points.\n\t\t#calculate_average_radius(circles : List[Circle]) -> float: Returns the average radius of all the circles.\n\t\t#nest_circles(circles : List[Circle], frame : Frame, step : float = 1, automatic : bool = False) -> Tuple[List[Circle], List[Circle]]: nests the circles, and returns 2 lists, one for the circle that got nested and the other for the circles that did not fit.\n\t\t\t--circles is a list of circles.\n\t\t\t--frame : is a Frame.\n\t\t\t--step is a float and defines by how much a circle is moved before checking if it fits.\n\t\t\t--automatic is a bool and it specifies if average step decrease is to be applied if a radius is smaller than average.\n"
        
        @staticmethod
        def usage() -> str:
            return "\nTo nest circles and store them in 2 lists and frame:\n\tnest, not_nest = Algorithm.nest_circles(circles, frame, 0.1) #nest = nested circles, not_nest = circles that did not fit.\nTo nest circles and store them in frame only:\n\tAlgorithm.nest_circles(circles, frame, 0.1) #nested circles are stored in frame only, not nested circles are not stored."

    @staticmethod
    def demo():
        circles = [
            Circle(29.25),
            Circle(29.25, 1),
            Circle(20),
            Circle(15),
            Circle(20),
            Circle(10),
            Circle(10),
            Circle(5),
            Circle(9),
            Circle(7),
            Circle(1),
            Circle(2)
        ]

        frame = Frame(100, 100)

        nest, not_nest = Algorithm.nest_circles(circles, frame, 0.1)

        frame.plot()
        print("Frame data:")
        print("\t" + frame.get_frame_data())

        print("Circles that got nested:")
        for n in nest:
            print("\t" + n.get_circle_data())

        print("Circles that did not get nested:")
        for n in not_nest:
            print("\t" + n.get_circle_data())

def Credit() -> None:
    string1 = "\nCredit for the Circular Nesting Technique:"
    string2 = "\n\t- Technique inspired by 'A simple solution for shape packing in 2D' by Ahmad Moussa on gorillasun.de."
    string3 = "\n\t- Python implementation of grid laying technique (moving one step and checking if it fits) by me: Raphael El Mouallem."
    
    print(string1, string2, string3)

#Help.demo()

#Credit()