import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple
import matplotlib.axes._axes as Axes
import random

class Rectangle:
    def __init__(self, width : float, height : float, thicknessXleft : float = 0, thicknessYtop : float = 0, thicknessXright : float = 0, thicknessYbottom : float = 0 , bounded : bool =  False, x : float = -1, y : float = -1) -> None:
        self.width : float = 0
        self.height : float = 0

        self.x : float = x
        self.y : float = y

        self.thickness : bool = False
        self.thicknessXleft : float = 0
        self.thicknessXright : float = 0
        self.thicknessYtop : float = 0
        self.thicknessYbottom : float = 0

        self.rotate : bool = False

        try:
            if isinstance(width, float) or isinstance(width, int) or isinstance(height, float) or isinstance(height, int):
                if height <= 0 or width <= 0:
                    raise ValueError("Dimensions must be a positive float/int")
                self.width : float = width
                self.height : float = height
                if (height > width):
                    self.rotate90()
        except ValueError as e:
            print(e)

        self.area : float = self.width * self.height
        
        try:
            if thicknessXleft != 0 and thicknessXright != 0 and thicknessYtop != 0 and thicknessYbottom != 0:
                if thicknessXleft > 0 and thicknessXright > 0 and thicknessYtop > 0 and thicknessYbottom > 0:
                    if (thicknessXleft + thicknessXright) <= self.width and (thicknessYtop + thicknessYbottom) <= self.height:
                        self.thicknessXleft = thicknessXleft
                        self.thicknessXright = thicknessXright
                        self.thicknessYtop = thicknessYtop
                        self.thicknessYbottom = thicknessYbottom
                        self.thickness = True
                    else:
                        raise ValueError("Sum of thicknesses exceeds width/height")
                else:
                    raise ValueError("Must input all the positive values of the thickness of every side")
        except ValueError as e:
            print(e)
        
        if(self.thickness):
            self.rectangles : List[Rectangle] = []
            self.bounded : bool = bounded

    def rotate90(self) -> None:
        temp = self.width
        self.width = self.height
        self.height = temp
        self.rotate = not self.rotate
    
    def set_position(self, x : float, y : float):
        self.x = x
        self.y = y

    def compare_to(self, rectangle : 'Rectangle') -> bool: return self.area > rectangle.area
    
    def get_rectangle_data(self) -> str:
        string1 = "\nwidth: " + str(self.width) + " , height: " + str(self.height)
        string2 = "\nx: " + str(self.x) + " , y: " + str(self.y) + " , rotated: " + str(self.rotate) + " , area: " + str(self.area)
        string3 = ""
        if(self.thickness):
            string3 = "\nleft thickness: " + str(self.thicknessXleft) + " , right thickness: " + str(self.thicknessXright)  + " , top thickness: " + str(self.thicknessYtop) + " , bottom thickness: " + str(self.thicknessYbottom)
            for r in self.rectangles:
                string3 += "\n" + str(len(self.rectangles)) + " rectangles are nested inside this rectangle:\n\t" + r.get_rectangle_data().replace("\n", "\n\t")
        
        return string1 + string2 + string3
    
    def plot_rectangle(self, ax : Axes) -> None:
        color = (random.random(), random.random(), random.random())
        x = self.x
        y = self.y
        width = self.width
        height = self.height

        if self.thickness:
            # Draw the colored rectangle representing the main body
            colored_rectangle_patch = patches.Rectangle((x, y), width, height, edgecolor='black', facecolor=color)
            ax.add_patch(colored_rectangle_patch)

            # Draw the white rectangle on top to simulate thickness
            white_rectangle_patch = patches.Rectangle((x + self.thicknessXleft, y + self.thicknessYbottom), width - self.thicknessXleft - self.thicknessXright, height - self.thicknessYtop - self.thicknessYbottom, edgecolor='white', facecolor='white')
            ax.add_patch(white_rectangle_patch)
        else:
            rectangle_patch = patches.Rectangle((x, y), width, height, edgecolor='black', facecolor=color)
            ax.add_patch(rectangle_patch)

        # Add text annotations
        ax.text(self.x + self.width / 2, self.y + self.height / 2, f"Width: {self.width}\nHeight: {self.height}\nRotation: {self.rotate}",ha='center', va='center', fontsize=5)

class Frame:
    def __init__(self, width : float, height : float, x : float, y : float, bounded : bool) -> None:
        self.width : float = 0
        self.height : float = 0

        self.x : float = x
        self.y : float = y

        self.area : float = self.width * self.height
        self.bounded : bool = bounded
        self.rectangles : List[Rectangle] = []

        try:
            if isinstance(width, float) or isinstance(width, int) or isinstance(height, float) or isinstance(height, int):
                if height < 0 or width < 0:
                    raise ValueError("Dimensions must be a positive float/int")
                self.width : float = width
                self.height : float = height
        except ValueError as e:
            print(e)

    def can_add(self, rectangle : 'Rectangle') -> bool:
        for i in range(2):
            if(self.width >= rectangle.width and self.height >= rectangle.height):
                return True
            rectangle.rotate90()
            
        return False

    def add(self, s : 'Rectangle') -> None:
        s.x = self.x
        s.y = self.y
        self.bounded = True

    def plot(self):
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)

        for rectangle in self.rectangles:
            rectangle.plot_rectangle(ax)

        ax.set_title('Nested Rectangles in Frame')
        ax.set_xlabel('Width')
        ax.set_ylabel('Height')
        ax.grid(True)
        
        plt.show()
        return ax

    def get_frame_data(self) -> str:
        string1 = ""
        for r in self.rectangles:
            string1 += r.get_rectangle_data()

        return string1

class Algorithm:
    def __init__(self, width : float, height : float, x : float, y: float) -> None:
        self.initialFrame : Frame

        self.topLeftX : float = x
        self.topLeftY : float = y

        self.spaceTaken : float = 0
        self.spaceLeft : float

        self.totalNbRectangles : int = 0
        self.addedNbRectangles : int = 0

        self.overFlow : int = 0

        self.notes : str = ""

        try:
            if isinstance(width, float) or isinstance(width, int) or isinstance(height, float) or isinstance(height, int):
                if height <= 0 or width <= 0:
                    raise ValueError("Dimensions must be a positive float/int")
                self.initialFrame : Frame = Frame(width, height, 0, 0, False)
                self.spaceLeft : float = width * height
                self.notes : str = "no notes"
        except ValueError as e:
            print(e)

        self.shelves : List[Frame] = []

        self.rectangles : List[Rectangle] = []
        self.newRectangles : List[Rectangle] = []

    @staticmethod
    def compare(r1 : Rectangle, r2 : Rectangle) -> bool:
        return r1.area > r2.area
    
    def set_frame(self, wid : float, hei: float, x1 : float, y1 : float, bounded : bool) -> None:
        self.initialFrame = Frame(wid, hei, x1, y1, bounded)

    def setRectangles(self, s : List[Rectangle]) -> List[Rectangle]:
        self.rectangles : List[Rectangle] = s
        self.rectangles = sorted(self.rectangles, key=lambda rectangle : (-rectangle.area, -max(rectangle.width, rectangle.height)))
        self.totalNbRectangles = len(self.rectangles)
        return self.rectangles

    def bounded_packing(self, frame : Frame):
        pickedRectangle : Rectangle = None
        indexPacking : int = 0

        for i, rect in enumerate(self.rectangles):
            currentRect = rect

            if(frame.can_add(currentRect)):
                #print("R CA")
                pickedRectangle = currentRect
                indexPacking = i
                break

        if(pickedRectangle == None):
            return
        else:
            frame.add(pickedRectangle)
            self.rectangles.pop(indexPacking)
            self.newRectangles.append(pickedRectangle)
            self.addedNbRectangles += 1
            self.spaceTaken += pickedRectangle.area

            if(pickedRectangle.thickness):
                wi = (pickedRectangle.width - pickedRectangle.thicknessXleft - pickedRectangle.thicknessXright)
                he = pickedRectangle.height - pickedRectangle.thicknessYbottom - pickedRectangle.thicknessYtop
                sort = Algorithm(pickedRectangle.width - pickedRectangle.thicknessXleft - pickedRectangle.thicknessXright, pickedRectangle.height - pickedRectangle.thicknessYbottom - pickedRectangle.thicknessYtop, pickedRectangle.thicknessXleft + pickedRectangle.x, pickedRectangle.thicknessYbottom + pickedRectangle.y)
                sort.set_frame(wi, he, pickedRectangle.thicknessXleft + pickedRectangle.x, pickedRectangle.thicknessYbottom + pickedRectangle.y, False)
                sort.setRectangles(self.rectangles)
                sort.run()
                self.newRectangles.extend(sort.get_new_rectangle())
                self.rectangles = sort.rectangles
                pickedRectangle.rectangles.extend(sort.get_new_rectangle())
                
            s3X = frame.x
            s3Y = frame.y + pickedRectangle.height
            s3width = pickedRectangle.width
            s3height = frame.height - pickedRectangle.height

            s4X = frame.x + pickedRectangle.width
            s4Y = frame.y
            s4width = frame.width - pickedRectangle.width
            s4height = frame.height

            s3 : Frame = Frame(s3width, s3height, s3X, s3Y, True)
            s4 : Frame = Frame(s4width, s4height, s4X, s4Y, True)

            if(s3.area > s4.area):
                self.bounded_packing(s3)
                self.bounded_packing(s4)
            else:
                self.bounded_packing(s4)
                self.bounded_packing(s3)

    def unbounded_packing(self, frame : Frame):
        if(len(self.rectangles) == 0): return

        pickedRectangle : Rectangle = None
        indexPacking = 0

        for i, rect in enumerate(self.rectangles):
            currentRect = rect

            if(currentRect.width < currentRect.height):
                currentRect.rotate90()
            if(frame.can_add(currentRect)):
                #print("P CA ", currentRect.width)
                pickedRectangle = currentRect
                indexPacking = i
                break
        
        if(pickedRectangle == None):
            return
        else:
            frame.add(pickedRectangle)
            self.rectangles.pop(indexPacking)
            self.newRectangles.append(pickedRectangle)
            self.addedNbRectangles += 1
            self.spaceTaken += pickedRectangle.area
            
            if(pickedRectangle.thickness):
                wi = (pickedRectangle.width - pickedRectangle.thicknessXleft - pickedRectangle.thicknessXright)
                he = pickedRectangle.height - pickedRectangle.thicknessYbottom - pickedRectangle.thicknessYtop
                sort = Algorithm(pickedRectangle.width - pickedRectangle.thicknessXleft - pickedRectangle.thicknessXright, pickedRectangle.height - pickedRectangle.thicknessYbottom - pickedRectangle.thicknessYtop, pickedRectangle.thicknessXleft + pickedRectangle.x, pickedRectangle.thicknessYbottom + pickedRectangle.y)
                sort.set_frame(wi, he, pickedRectangle.thicknessXleft + pickedRectangle.x, pickedRectangle.thicknessYbottom + pickedRectangle.y, False)
                sort.setRectangles(self.rectangles)
                sort.run()
                self.newRectangles.extend(sort.get_new_rectangle())
                self.rectangles = sort.rectangles
                pickedRectangle.rectangles.extend(sort.get_new_rectangle())

            s1X = frame.x
            s1Y = frame.y + pickedRectangle.height
            s1width = pickedRectangle.width
            s1height = frame.height - pickedRectangle.height

            s2X = frame.x + pickedRectangle.width
            s2Y = frame.y
            s2width = frame.width - pickedRectangle.width
            s2height = frame.height

            s1 : Frame = Frame(s1width, s1height, s1X, s1Y, False)
            s2 : Frame = Frame(s2width, s2height, s2X, s2Y, True)

            s : Frame = s1

            self.bounded_packing(s2)
            self.unbounded_packing(s)

    def get_new_rectangle(self) -> List[Rectangle]: return self.newRectangles

    def get_unnested_rectangle(self) -> List[Rectangle]: return self.rectangles

    def run(self) -> None:
        self.unbounded_packing(self.initialFrame)
        self.overFlow = self.totalNbRectangles - self.addedNbRectangles

        if(self.overFlow > 0):
            self.notes = str(self.overFlow) + " rectangles haven't been nested"
        else:
            self.notes = "All rectangles got nested"
        
        self.initialFrame.rectangles = self.newRectangles

    def plot(self) -> None:
        return self.initialFrame.plot()

    def get_data(self) -> str: return self.initialFrame.get_frame_data()

class Help:
    class rectangle_class_help:
        @staticmethod
        def rectangle_definition() -> str:
            string1 = "\nRectangle class Documentation:\n\t-Utility: Represents a rectangle with a given width an d height and optional thickness for each of its 4 sides."
            string2 = "\n\t-Fields:\n\t\t#width and height: always positive or else value error will be raised.\n\t\t#thicknessXleft, thicknessXright, thicknessYtop, thicknessYbottom: (Optional) represent a homogeneous thickness for each side of the rectangle, rendering it similar to a frame with a rectangular empty area in the middle, thicknesses are always positive, the sum of the x-thickness <= width and the sum of the y-thickness <= height, or else value error will be raised.\n\t\t#thickness: boolean represents if thicknesses exist.\n\t\t#rotate: boolean represents if the rectangle have been rotated.\n\t\t#rectangles: List of rectangles that have been nested inside this rectangles frame, if the rectangle have thicknesses.\n\t\t#area: width*height of the rectangle."
            string3 = "\n\t-Methods:\n\t\t#rotate90(self) -> None: rotates rectangle 90 deg.\n\t\t#get_rectangle_data(self) -> str: Returns the rectangles field values in a string.\n\t\t#plot_rectangle(self, ax : Axes) -> None: Add the rectangle to the axe of a matplotlib orthogonal axe.\n\t\t#compare_to(self , rectangle : 'Rectangle') -> bool: compare a another rectangle with this rectangle according to area, returns False if the other rectangle has greater area.\n"

            return string1 + string2 + string3

        @staticmethod
        def usage() -> str:
            string1 = "\nTo create a single rectangle:\n\trectangle = Rectangle(50, 50.5) #where width and height are positive ints/floats.\n\trectangle = Rectangle(50, 50.5, 1, 2.5, 3, 10) #where thicknesses of each side of the rectangle are positive ints/floats."
            string2 = "\nTo create a list of recangles:\n\trectangles = [\n\t\tRectangle(10.5, 5),\n\t\tRectangle(10, 15),\n\t\tRectangle(20, 10),\n\t\tRectangle(50, 50, 10, 20, 30, 10),\n\t\tRectangle(50, 5)\n\t]"

            return string1 + string2
    
    class frame_class_help:
        @staticmethod
        def frame_definition() -> str:
            string1 = "\nFrame class Documentation:\n\t-Utility: Represents a frame with a given width and height to nest the rectangles in."
            string2 = "\n\t-Fields:\n\t\t#width & height: Represent the width and height of the frame, always positive or else value error will be raised.\n\t\t#x & y: represents the bottom left position of the frame for recursive nesting.\n\t\t#area: width * height.\n\t\t#rectangles: list of the rectangles that got nested in this frame.\n\t\t#bounded: boolean represents if the frame is bounded or not."

            return string1 + string2

        @staticmethod
        def usage() -> str:
            return "No need to use it, Algorithm class takes care of it all."

    class algorithm_class_help:
        @staticmethod
        def algorithm_definition() -> str:
            return "\nAlgorithm class Document:\n\t-Utility: This class contains multiple functions that helps the main function run() to nest rectangles using bounding recursive technique.\n\t-Methods:\n\t\t#run(self) -> None: nests the rectangles.\n\t\t#get_new_rectangle(self) -> List[Rectangle]: returns the nested rectangles.\n\t\t#get_unnested_rectangle(self) -> List[Rectangle]: Returns the rectangles that did not get nested.\n\t\t#plot(self) -> None: plots the nested rectangles within the boundaries of the frame.\n\t\t#get_data(self) -> str: Returns a string of rectangles data.\n\t\t#setRectangles(self, s : List[Rectangle]) : insert a list of rectangles for the algorithm to nest."

        @staticmethod
        def usage() -> str:
            return "\nTo nest rectangles:\n\tnest = Algorithm(50, 80, 0, 0)\n\trectangles = [\n\t\tRectangle(10.5, 5),\n\t\tRectangle(10, 15),\n\t\tRectangle(20, 10),\n\t\tRectangle(50, 50, 10, 20, 30, 10),\n\t\tRectangle(50, 5)\n\t]\n\tnest.setRectangles(rectangles)\n\tnest.run()\n\tnest.plot()"

    @staticmethod
    def demo():
        nest = Algorithm(50, 80, 0, 0)

        rectangles = [
            Rectangle(10,5),
            Rectangle(10,15),
            Rectangle(20,10),
            Rectangle(50,50, 10, 20, 30, 10),
            Rectangle(50,5),
            Rectangle(10,5),
            Rectangle(10,5),
            Rectangle(10, 20),
            Rectangle(50,50, 10, 20, 30, 10),
            Rectangle(50,5),
            Rectangle(10,5),
            Rectangle(10, 20),
            Rectangle(40, 5),
        ]

        #rectangles = sorted(rectangles, key=lambda rectangle : rectangle.area, reverse=True)

        nest.setRectangles(rectangles)

        nest.run()

        print("Nested Rectangles:\n" , nest.get_data())

        not_rects = nest.get_unnested_rectangle()

        print("\nUnnested rectangles:")
        for n in not_rects:
            print(n.get_rectangle_data())

        nest.plot()

def Credit() -> None:
    string1 = "\nCredit for the Rectangular Nesting Algorithm:"
    string2 = "\n\t- Algorithm by Defu Zhanga, Yan Kangb and Ansheng Denga, described in 'A new heuristic recursive algorithm for the strip rectangular packing problem'."
    string3 = "\n\t- Java implementation by Elie Atamech, former student from the Lebanese University, Fanar Faculty of Sciences, Branch 2."
    string4 = "\n\t- Python implementation and refinement by extending the algorithm to support nesting within frame rectangles—rectangles with varying thicknesses along each side. This enhancement allowed for intricate nesting configurations within these specially defined shapes. Developed and fine-tuned by me: Raphael El Mouallem."

    print(string1 + string2 + string3 + string4)

#Help.demo()

#Credit()