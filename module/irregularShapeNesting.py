import matplotlib.pyplot as plt
import matplotlib.path as mpath
import numpy as np
import cv2
from typing import List, Tuple, Optional, Union
import pymunk
import math
import random

class Polygon:
    def __init__(self, vertices : List[np.array]):
        self.original_vertices : List[np.array] = np.array(vertices)
        self.polygon : List[np.array] = np.array(vertices)
        self.simplifying_eps : float
    
    def plot(self) ->  None:
        if isinstance(self.polygon, list):
            self.polygon = np.array(self.polygon)

        x = self.polygon[:, 0, 0]
        y = self.polygon[:, 0, 1]

        plt.figure()
        plt.plot(x, y, 'b-')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Polygon')
        plt.grid(True)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

    def simplify(self, epsilon : float = 1.2) -> List[np.array]:
        # Convert vertices to numpy array
        polygon_np = np.array(self.polygon, dtype=np.int32)
        
        # Reshape polygon array for cv2.approxPolyDP function
        polygon_np = polygon_np.reshape((-1, 1, 2))
        
        try:
            # Simplify polygon using Douglas-Peucker algorithm
            simplified_polygon = cv2.approxPolyDP(polygon_np, epsilon, True)
            
            # Convert simplified polygon back to list format
            simplified_polygon = simplified_polygon.squeeze()
            
            # Append the first point of the simplified polygon to itself to close the loop
            simplified_polygon = np.expand_dims(simplified_polygon, axis=1)
            simplified_polygon = np.vstack([simplified_polygon, [simplified_polygon[0]]])

            self.polygon = simplified_polygon
            self.simplifying_eps = epsilon
            
            return simplified_polygon
        
        except Exception as e:
            print("An error occurred while simplifying the polygon:", e)
            return None

    def area(self) -> float:
        try:
            area = 0.0
            for i in range(len(self.polygon)):
                j = (i + 1) % len(self.polygon)
                area += self.polygon[i][0][0] * self.polygon[j][0][1]
                area -= self.polygon[j][0][0] * self.polygon[i][0][1]
            area = abs(area) / 2.0
            return area
        except Exception as e:
            print(f"An error occurred while calculating the area: {e}")
            return None

    def rectangular_area(self) -> float:
        try:
            # Find the minimum and maximum x and y coordinates of the vertices
            min_x = min(self.polygon[:, 0, 0])
            max_x = max(self.polygon[:, 0, 0])
            min_y = min(self.polygon[:, 0, 1])
            max_y = max(self.polygon[:, 0, 1])
        
            # Calculate the width and height of the bounding box
            width = max_x - min_x
            height = max_y - min_y
        
            # Calculate the rectangular area
            area = width * height
        
            return area
        except Exception as e:
            print(f"An error occurred while calculating rectangular area: {e}")
            return None

    def centroid(self) -> Tuple[float, float] :
        try:
            self.shift_polygon_to_origin()
            vertices_array = np.array(self.polygon)
            # Flatten the vertices array to simplify calculations
            flat_vertices = vertices_array.reshape(-1, 2)
            # Convert vertices to integer type (required by cv2.moments)
            flat_vertices_int = np.round(flat_vertices).astype(int)
            # Create a mask image from the vertices
            mask = np.zeros((1000, 1000), dtype=np.uint8)
            cv2.drawContours(mask, [flat_vertices_int], -1, (255), thickness=cv2.FILLED)

            # Calculate moments
            moments = cv2.moments(mask)

            # Calculate centroid
            centroid_x = moments['m10'] / moments['m00']
            centroid_y = moments['m01'] / moments['m00']

            return centroid_x, centroid_y
        
        except Exception as e:
            print(f"An error occurred while calculating centroid: {e}")
            return None, None

    def translate_to_point(self, target_point : Tuple[float, float] = [0, 0]) -> List[np.array]:
        try:
            centroid_x, centroid_y = self.centroid()
            translation_vector = np.array(target_point) - np.array([centroid_x, centroid_y])
            translated_vertices = self.polygon + translation_vector.reshape(1, 1, 2)
            self.polygon = translated_vertices
            return translated_vertices
        except Exception as e:
            print(f"An error occurred while translating the polygon: {e}")
            return None

    def resize_ratio(self, ratio : float) -> List[np.array]:
        try:
            centroid_x, centroid_y = self.centroid()
            # Translate the polygon vertices so that the centroid is at the origin
            translated_polygon = self.polygon - [centroid_x, centroid_y]
            # Resize the translated polygon
            self.polygon = translated_polygon * ratio
            # Translate the resized polygon back to its original position
            resized_polygon = self.translate_to_point([centroid_x, centroid_y])
            self.polygon = resized_polygon
            return resized_polygon
        except Exception as e:
            print(f"An error occurred while resizing the polygon: {e}")
            return None

    def resize(self, new_width : float, new_height : float) -> List[np.array]:
        try:
            polygon = self.polygon
            centroid_x, centroid_y = self.centroid()
            translated_polygon = polygon - [centroid_x, centroid_y]
            min_x = np.min(translated_polygon[:, 0, 0])
            max_x = np.max(translated_polygon[:, 0, 0])
            min_y = np.min(translated_polygon[:, 0, 1])
            max_y = np.max(translated_polygon[:, 0, 1])
            current_width = max_x - min_x
            current_height = max_y - min_y
            scale_x = new_width / current_width
            scale_y = new_height / current_height
            resized_polygon = translated_polygon * [scale_x, scale_y]
            resized_polygon += [centroid_x, centroid_y]
            self.polygon = resized_polygon
            return resized_polygon
        except Exception as e:
            print(f"An error occurred while resizing the polygon: {e}")
            return None

    def point_in_polygon(self, point : Tuple[float, float]) -> Union[bool, None]:
        try:
            if isinstance(point, (list, np.ndarray)):
                point = point[0]


            polygon_copy = self.polygon.copy()

            if isinstance(polygon_copy, list) and all(isinstance(arr, np.ndarray) for arr in polygon_copy):
                polygon_copy = polygon_copy[0]

            poly = polygon_copy.squeeze()
            path = mpath.Path(poly)
            return path.contains_point(point)
        except Exception as e:
            print(f"An error occurred while checking point in polygon: {e}")
            return None
    
    def point_in_polygon_ray_cast(self, point) -> Union[bool, None]:
        try:
            x, y = point
            n = len(self.polygon)
            inside = False
            p1x, p1y = self.polygon[0][0]

            for i in range(n + 1):
                p2x, p2y = self.polygon[i % n][0]
                if y > min(p1y, p2y):
                    if y <= max(p1y, p2y):
                        if x <= max(p1x, p2x):
                            if p1y != p2y:
                                xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                            if p1x == p2x or x <= xints:
                                inside = not inside
                p1x, p1y = p2x, p2y
            return inside
        except Exception as e:
            print(f"An error occurred while checking point in polygon: {e}")
            return None

    def rotate(self, angle : float) -> Union[List[np.array], None]:
        try:
            centroid_x, centroid_y = self.centroid()
            translated_vertices = self.polygon - np.array([centroid_x, centroid_y])
            angle_rad = np.radians(angle)
    
            rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],[np.sin(angle_rad), np.cos(angle_rad)]])
    
            rotated_vertices = np.dot(translated_vertices, rotation_matrix.T)
            final_vertices = rotated_vertices + np.array([centroid_x, centroid_y])
        
            self.polygon = final_vertices
    
            return final_vertices
        except Exception as e:
            print(f"An error occurred while rotating the polygon: {e}")
            return None
    
    def mirror(self):
        flat_polygon = [coord for sublist in self.polygon for coord in sublist]

        # Find the maximum x-coordinate
        image_width = max(coord[0] for coord in flat_polygon)

        # Mirror each point by subtracting its x-coordinate from the max
        mirrored_polygon = [[image_width - coord[0], coord[1]] for coord in flat_polygon]
        reshaped_polygon = [[coord] for coord in mirrored_polygon]
        self.polygon = reshaped_polygon
        return mirrored_polygon

    def draw_polygon(self, ax: plt.Axes, frame : bool = False) -> None:
        if(frame):
            color = (1, 1, 1)
        else:
            color = (random.random(), random.random(), random.random())

        # Extract x and y coordinates
        x = self.polygon[:, 0, 0]
        y = self.polygon[:, 0, 1]

        # Create and add the polygon patch
        ax.fill(x, y, edgecolor='black', facecolor=color, linestyle='-')

    @staticmethod 
    def calculate_edge_vector(vertex1 : List[np.array], vertex2 : List[np.array]) -> List[int]:
        return [vertex2[0][0] - vertex1[0][0], vertex2[0][1] - vertex1[0][1]]

    @staticmethod
    def normalize_vector(vector : List[np.array]) -> List[int]:
        magnitude = sum(x**2 for x in vector)**0.5
        if magnitude < 1e-10:  # Threshold for near-zero magnitude
            return [0.0, 0.0]  # Return zero vector if magnitude is negligible
        else:
            return [x / magnitude for x in vector]

    @staticmethod
    def calculate_angle(vector1, vector2):
        dot_product = sum(a * b for a, b in zip(vector1, vector2))
        angle = math.acos(dot_product)
        return math.degrees(angle)  # Convert to degrees if needed

    @staticmethod
    def identify_tip(self, previous_vector, current_vertex, next_vertex, threshold_angle):
        # Special case for last vertex (avoid comparing with first vertex)
        if all(a == b for a, b in zip(current_vertex[0], next_vertex[0])):
            return False

        edge_vector = self.calculate_edge_vector(current_vertex, next_vertex)
        unit_vector = self.normalize_vector(edge_vector)

        # Check if first edge or angle exceeds threshold
        return (previous_vector is None or self.calculate_angle(previous_vector, unit_vector) <= threshold_angle)

    def find_tips(self, threshold_angle : float = 120) -> List[np.ndarray]:
        tips = []
        previous_vector = None

        for i in range(len(self.polygon)):
            current_vertex = self.polygon[i]
            next_vertex = self.polygon[(i + 1) % len(self.polygon)]  # Wrap around for last edge

            if self.identify_tip(self, previous_vector, current_vertex, next_vertex, threshold_angle):
                tips.append(current_vertex)

            previous_vector = self.normalize_vector(self.calculate_edge_vector(current_vertex, next_vertex))

        return tips
  
    def get_x_values_for_y(self, y_value : float) -> List[float]:
        x_values : List[float] = []

        for i in range(len(self.polygon)):
            p1 = self.polygon[i][0]
            p2 = self.polygon[(i + 1) % len(self.polygon)][0]  # Wrap around to the first point

            if p1[1] != p2[1]:  # Ensure the edge is not horizontal
                if min(p1[1], p2[1]) <= y_value <= max(p1[1], p2[1]):  # Check if y_value is within y-range of edge
                    # Interpolate x-value for given y-value
                    x = p1[0] + (y_value - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1])
                    x_values.append(x)

        return x_values

    def equal_polygons(self, poly : 'Polygon') -> bool:
        polygon = poly.polygon
        # Check if the number of vertices is the same
        if len(self.polygon) != len(polygon):
            return False
    
        sorted_poly1 = np.sort(self.polygon, axis=0)
        sorted_poly2 = np.sort(polygon, axis=0)

        return np.array_equal(sorted_poly1, sorted_poly2)

    def exact_polygons(self, poly : 'Polygon') -> bool:
        polygon = poly.polygon
        # Check if the number of vertices is the same
        if len(self.polygon) != len(polygon):
            return False
    
        # Check if all vertices are the same
        for vertex1, vertex2 in zip(self.polygon, polygon):
            if (vertex1 != vertex2).any():
                return False
    
        return True

    def shift_polygon_to_origin(self) -> Union[List[np.array], None]:
        try:
            polygon = np.array(self.polygon)
        
            min_x = np.min(polygon[:, 0, 0])
            min_y = np.min(polygon[:, 0, 1])
        
            # Shift all coordinates by the minimum values
            shifted_polygon = polygon - [min_x, min_y]
            self.polygon = shifted_polygon
            return shifted_polygon
        
        except Exception as e:
            print(f"An error occurred while shifting the polygon to origin: {e}")
            return None

class ComputerVision:
    def __init__(self, image_path : str, threshold : int = 127 ,eps : float = 0.00000000000000001) -> None:
        self.image_path : str = image_path
        self.image : Union[np.ndarray, None] = self.read_image(self.image_path)
        self.binary_image : Union[np.ndarray, None] = self.convert_to_binary(self.image, threshold)

        self.contour : List[np.ndarray]
        self.hierarchy : np.ndarray

        self.contour , self.hierarchy = self.detect_shapes(self.binary_image)
        self.polygons : List[np.ndarray] =  self.approximate_polygons(self.contour, eps)

        self.polygons, self.hierarchy = self.clean_polygons(self.polygons, self.hierarchy, self.binary_image)

    def get_polygon_list(self, polygons : List[np.array] = None) -> List[Polygon]:
        if polygons is None:
            polygons = self.polygons
        polygon_list: List[Polygon] = []
        for polygon in polygons:
            polygon_list.append(Polygon(polygon))
        return polygon_list

    @staticmethod
    def read_image(image_path : str) -> Union[np.ndarray, None]:
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        try:
            if image is None:
                print("Error: Failed to read the image.")
                return None

        except Exception as e:
            print("An error occurred while reading the image:", e)
            return None
        
        return image
    
    @staticmethod
    def convert_to_binary(image : Union[np.ndarray, None], threshold : int =127) -> Union[np.ndarray, None]:
        if image is None:
            print("Error: Image is None.")
            return None

        try:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, binary_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)

        except Exception as e:
            print("An error occurred while converting the image to binary:", e)
            return None
    
        return binary_image
    
    @staticmethod
    def detect_shapes(binary_image : Union[np.ndarray, None]) ->  Tuple[List[np.ndarray], np.ndarray]:
        try:
            if binary_image is None:
                raise ValueError("Input binary image is None.")

            contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            detected_contours = []
            for contour in contours:
                detected_contours.append(contour)

            return detected_contours, hierarchy

        except Exception as e:
            print("An error occurred while detecting shapes:", e)
            return [], None

    @staticmethod
    def approximate_polygons(contours : List[np.ndarray], eps : float = 0.00000000000000001) -> List[np.ndarray]:
        try:
            approximated_polygons = []

            for contour in contours:
                epsilon = eps * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                approx_closed = np.concatenate((approx, [approx[0]]))
                approximated_polygons.append(approx_closed)

            return approximated_polygons

        except Exception as e:
            print("An error occurred while approximating polygons:", e)
            return []

    @staticmethod
    def remove_polygon_by_index(polygons : List[np.ndarray], hierarchy : np.ndarray, index : int) -> Tuple[List[np.ndarray], np.ndarray]:
        try:
            if index < 0 or index >= len(polygons):
                print("Error: Invalid index.")
                return polygons, hierarchy
        
            del polygons[index]

            next = hierarchy[0][index][0]
            prev = hierarchy[0][index][1]
            chil = hierarchy[0][index][2]
            pare = hierarchy[0][index][3]
        
            # Remove the corresponding entry from the hierarchy array
            hierarchy = np.delete(hierarchy, index, axis=1)
        
            # Update the hierarchy array to adjust parent indices
            for i in range(len(hierarchy[0])):
                if(hierarchy[0][i][3] == index):
                    hierarchy[0][i][3] = pare
        
            return polygons, hierarchy

        except Exception as e:
            print("An error occurred while removing polygon by index:", e)
            return polygons, hierarchy

    def clean_polygons(self, polygons : List[np.ndarray], hierarchy : np.ndarray, image : Union[np.ndarray, None]) -> Tuple[List[np.ndarray], np.ndarray]:
        try:
            image_height = image.shape[0]
            image_width = image.shape[1]

            cleaned_polygons = []

            #x, y, w, h = cv2.boundingRect(polygons[0])
            polygon = polygons[0]
            max_x = np.max(polygon[:, 0, 0])
            min_x = np.min(polygon[:, 0, 0])
            max_y = np.max(polygon[:, 0, 1])
            min_y = np.min(polygon[:, 0, 1])

            # Calculate width and height
            w = max_x - min_x
            h = max_y - min_y

            polygon_area = cv2.contourArea(polygons[0])
            image_area = image_width * image_height

            threshold = 0.001

            # Check if the polygon matches the size of the image frame
            if (w >= (image_width - (image_width * threshold)) and h >= image_height - ((image_height * threshold))) or (polygon_area >= (image_area - (image_area * threshold))) or (w > image_width or h > image_height) or ((w == image_width - 1 and h == image_height - 1)):
                cleaned_polygons, hierarchy = self.remove_polygon_by_index(polygons, hierarchy, 0)
                return cleaned_polygons, hierarchy
            else:
                return polygons, hierarchy

        except Exception as e:
            print("An error occurred while cleaning polygons:", e)
            return polygons, hierarchy
        
    def plot_result(self, approximated_polygons : List[np.ndarray] = None, binary_image : Union[np.ndarray, None] = None) -> None:
        if approximated_polygons is None:
            approximated_polygons = self.polygons
        if binary_image is None:
            binary_image = self.binary_image
        if binary_image is None:
            print("Error: Binary image is None.")
            return

        # Create a blank image
        polygon_img = np.ones_like(binary_image) * 255

        try:
            # Draw polygons on the blank image
            cv2.polylines(polygon_img, approximated_polygons, isClosed=True, color=(0, 0, 255), thickness=1)
        except Exception as e:
            print("An error occurred while drawing polygons on the image:", e)
            return

        # Display the result
        try:
            plt.figure(figsize=(10, 8))
            plt.imshow(cv2.cvtColor(polygon_img, cv2.COLOR_BGR2RGB))
            plt.title('Polygons Detected')
            plt.axis('off')
            plt.show()
        except Exception as e:
            print("An error occurred while displaying the result:", e)

    @staticmethod
    def merge_polygons(polygon1 : List[np.array], polygon2 : List[np.array]) -> List[np.array]:
        merged_polygon = np.vstack((polygon1, polygon2))
        connecting_line = np.array([polygon2[-1], polygon1[0]])
        merged_polygon = np.vstack((merged_polygon, connecting_line))
        return merged_polygon

    #check why polygons are going missing
    def merge_polygons_by_hierarchy(self) -> Union[List[np.array], None]:
        try:
            merged_polygons = []
            hierarchy_copy = self.hierarchy.copy()

            def arrange_hierarchy(h, index):
                if index < 0 or index >= len(h):
                    print("Error: Invalid index.")
                    return h

                pare = h[0][index][3]
            
                # Remove the corresponding entry from the hierarchy array
                h = np.delete(h, index, axis=1)
            
                # Update the hierarchy array to adjust parent indices
                for i in range(len(h[0])):
                    if(h[0][i][3] == index):
                        h[0][i][3] = pare
                
                return h

            i = 0
            length = len(hierarchy_copy[0])
            while i < length:
                if hierarchy_copy[0][i][3] == -1:  # Check if contour has no parent
                    poly = self.polygons[i]
                    tempi = i
                    tempj = []

                    j = i
                    while j < length:
                        if hierarchy_copy[0][j][3] == i:
                            poly = self.merge_polygons(poly, self.polygons[j])
                            tempj.append(j)

                        j += 1

                    merged_polygons.append(poly)
                    length -= 1 + len(tempj)
                
                    hierarchy_copy = arrange_hierarchy(hierarchy_copy, tempi)
                    
                    c = 1
                    for j in tempj:
                        hierarchy_copy = arrange_hierarchy(hierarchy_copy, j - c)
                        c += 1
                
                i += 1
            
            return merged_polygons
        except Exception as e:
            print(f"An error occurred while merging polygons by hierarchy: {e}")
            return None

    """
    def merge_polygons_by_hierarchy(self) -> Union[List[np.array], None]:
        try:
            merged_polygons = []
            hierarchy_copy = self.hierarchy.copy()

            def arrange_hierarchy(h, index):
                if index < 0 or index >= len(h):
                    print("Error: Invalid index.")
                    return h

                pare = h[0][index][3]
            
                # Remove the corresponding entry from the hierarchy array
                h = np.delete(h, index, axis=1)
            
                # Update the hierarchy array to adjust parent indices
                for i in range(len(h[0])):
                    if(h[0][i][3] == index):
                        h[0][i][3] = pare
                
                return h

            i = 0
            length = len(hierarchy_copy[0])
            while i < length:
                if hierarchy_copy[0][i][3] == -1:  # Check if contour has no parent
                    poly = self.polygons[i]
                    tempi = i
                    tempj = []

                    j = i
                    while j < length:
                        if hierarchy_copy[0][j][3] == i:
                            poly = self.merge_polygons(poly, self.polygons[j])
                            tempj.append(j)

                        j += 1

                    merged_polygons.append(poly)
                    length -= 1 + len(tempj)
                    
                    hierarchy_copy = arrange_hierarchy(hierarchy_copy, tempi)
                    
                    c = 1
                    for j in tempj:
                        hierarchy_copy = arrange_hierarchy(hierarchy_copy, j - c)
                        c += 1
                    
                    # Update the loop counter to skip merged indices
                    i += len(tempj)
                
                i += 1
            
            self.polygon_list = self.__create_polygon_list(merged_polygons)
            return merged_polygons
        except Exception as e:
            print(f"An error occurred while merging polygons by hierarchy: {e}")
            return None
    """

class Algorithm:
    def __init__(self, polygons : List[Polygon], frame : Polygon) -> None:
        self.polygons : List[Polygon] = polygons
        self.frame : Polygon = frame

        self.nested_polygons : List[Polygon]
        self.not_nested_polygons : List[Polygon]
        self.nesting_results : List[Polygon]

    def get_nesting_results(self) -> List[Polygon]:
        self.nesting_results = self.nested_polygons.copy()  # Create a copy to avoid modifying original list
        self.nesting_results.extend([self.frame])
        return self.nesting_results
    
    def plot_results(self) -> None:
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.set_xlim(0, max(self.frame.polygon[:, 0, 0]))
        ax.set_ylim(0, max(self.frame.polygon[:, 0, 1]))
        
        self.frame.draw_polygon(ax, True)

        for p in self.nested_polygons:
            p.draw_polygon(ax)

        ax.set_title('Nested polygons in Frame')
        ax.set_xlabel('Width')
        ax.set_ylabel('Height')
        ax.grid(True)
                
        plt.show() 

    def shift_polygons_to_origin(self) -> List[Polygon]:
        shifted_polygons : List[Polygon] = []
        for polygon in self.polygons:
            polygon.shift_polygon_to_origin()
            shifted_polygons.append(polygon)
        
        self.polygons = shifted_polygons
        return shifted_polygons

    def sort_polygons_by_area(self) -> List[Polygon]:
        self.polygons = sorted(self.polygons, key=lambda polygon: polygon.rectangular_area(), reverse=True)
        return self.polygons
    
    def polygon_in_frame(self, polygon : Polygon) -> bool:
        # Check if all vertices of the polygon are within the frame
        f = self.frame

        for vertex in polygon.polygon:
            if not f.point_in_polygon(vertex):
                return False
        
        for vertex in f.polygon:
            if polygon.point_in_polygon(vertex):
                return False

        return True

    @staticmethod
    def collision_detection(polygon1 : Polygon, polygon2 : Polygon) -> Union[bool, None]:
        #try:
        """for vertex in polygon1:
            if point_in_polygon(vertex, polygon2):
                return True
        for vertex in polygon2:
            if point_in_polygon(vertex, polygon1):
                return True"""
                
        space = pymunk.Space()
        space.gravity = (0, 0)  # No gravity

        # Convert polygons to Pymunk format
        poly1_pymunk = [tuple(vertex[0]) for vertex in polygon1.polygon]
        poly2_pymunk = [tuple(vertex[0]) for vertex in polygon2.polygon]

        # Create bodies for polygons
        body1 = pymunk.Body()
        body2 = pymunk.Body()

        # Create PyMunk polygons for polygon1
        poly1_shape = pymunk.Poly(body1, poly1_pymunk)
        poly1_shape.color = (255, 0, 0)  # Color of polygon1 for visualization
        space.add(body1, poly1_shape)

        # Create PyMunk polygons for polygon2
        poly2_shape = pymunk.Poly(body2, poly2_pymunk)
        poly2_shape.color = (0, 255, 0)  # Color of polygon2 for visualization
        space.add(body2, poly2_shape)

        # Check for collisions
        collisions = space.shape_query(poly1_shape) or space.shape_query(poly2_shape)

        if (collisions):
            return True

        return False
        """
        except Exception as e:
            print(f"An error occurred during collision detection: {e}")
            return None"""

    def nest_polygons(self, rotating : bool = True, mirroring : bool = True, step : float = 1):
        frame_polygon = self.frame.shift_polygon_to_origin()
        frame_squeezed = np.array(frame_polygon)
        frame_squeezed = frame_squeezed.squeeze()

        #self.frame.plot()

        sorted_polygons = self.sort_polygons_by_area()
        self.nested_polygons : List[Polygon] = []
        self.not_nested_polygons : List[Polygon] = []

        for polygon in sorted_polygons:
            # Translate polygon to origin
            polygon.shift_polygon_to_origin()
            shifted_polygon = polygon
                    
            polygon.shift_polygon_to_origin()
            
            max_x = int(frame_squeezed[:, 0].max())
            max_y = int(frame_squeezed[:, 1].max())

            minX = min(polygon.polygon[:, 0, 0])
            maxX = max(polygon.polygon[:, 0, 0])
            minY = min(polygon.polygon[:, 0, 1])
            maxY = max(polygon.polygon[:, 0, 1])
        
            # Calculate the width and height of the bounding box
            width = maxX - minX
            height = maxY - minY

            width = int(width / 2)
            height = int(height / 2)
            
            nestable = True
            poly : Polygon = shifted_polygon
            pol : List[np.array]

            y = height
            while(y < max_y - height):
                y += step

            #for y in range(height, max_y - height, step):
                nestable = False
                pts = self.frame.get_x_values_for_y(y)
                x1 = 0
                x2 = 0
                if pts:
                    if(pts[0] is not None): x1 = int(min(pts))
                    if(pts[-1] is not None): x2 = (max_x - int(max(pts)) - 1)

                if(x1 == 0): x1 = 2 * width
                if(x2 == -1): x2 = 2 * width

                x = x1 - width
                while(x < max_x + width - x2):
                    x += step
                
                #for x in range(x1 - width, max_x + width - x2, step):
                    nestable = True
                    poly.translate_to_point([x, y])
                    angle = 0
                    mirror = False

                    temp = poly
                    if(rotating):
                        while(angle >= 0 and angle < 360):
                            nestable = True
                            poly = temp
                            if (angle > 0) :
                                poly.rotate(angle)
                            if(mirror and mirroring):
                                poly.mirror()
                                poly.translate_to_point([x, y])
                            if self.polygon_in_frame(poly):
                                if len(self.nested_polygons) > 0:
                                    for nested in self.nested_polygons:
                                        collision = self.collision_detection(nested, poly)
                                        if collision:
                                            nestable = False
                                            break
                            else:
                                nestable = False

                            if nestable:
                                break
                            else:
                                if(not mirror and mirroring):
                                    mirror = not mirror
                                    continue
                                else:
                                    angle += 1
                                    mirror = False

                    if self.polygon_in_frame(poly):
                        if len(self.nested_polygons) > 0:
                            for nested in self.nested_polygons:
                                collision = self.collision_detection(nested, poly)
                                if collision:
                                    nestable = False
                                    break
                    else:
                        nestable = False
                        continue

                    if nestable:
                        break

                if nestable:
                    break

            if nestable:
                print("Polygon nested")
                self.nested_polygons.append(poly)
                continue
            else:
                print("Polygon not nested")
                self.not_nested_polygons.append(poly)
        
        return self.nested_polygons, self.not_nested_polygons

class Help:
    def demo():
        c = ComputerVision("screen.png", 230, 0.001)

        l = c.get_polygon_list()

        for n in l:
            n.resize_ratio(0.3)
            n.shift_polygon_to_origin()
        l[0].resize_ratio(5)

        l2 = l[6:]

        a = Algorithm(l2, l[0])
        a.nest_polygons()
        a.plot_results()
        pass

def Credit() -> None:
    print("This module was created by me: Raphael El Mouallem as a part of my final year project in 2024 for my computer science bachelor at the lebanese university of sciences section II Fanar.")

Help.demo()

Credit()