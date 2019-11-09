import cv2
import numpy as np


class QVisualisation:
    def __init__(self, magnification):
        # Set the magnification factor of the display
        self.magnification = magnification
        # Set the initial state of the goal
        self.goal_state = np.array([0.75, 0.85], dtype=np.float32)
        # Set the width and height of the environment
        self.width = 1.0
        self.height = 1.0
        # Create an image which will be used to display the environment
        self.image = np.zeros([int(self.magnification * self.height), int(self.magnification * self.width), 3], dtype=np.uint8)
        self.image.fill(0)
        # Create base triangles for each square
        self.top_triangle = np.array([[0, 0], [0.1, 0], [0.05, 0.05]])
        self.bottom_triangle = np.array([[0.1, 0.1], [0, 0.1], [0.05, 0.05]])
        self.right_triangle = np.array([[0.1, 0.1], [0.1, 0], [0.05, 0.05]])
        self.left_triangle = np.array([[0, 0], [0, 0.1], [0.05, 0.05]])
        # Create offset triangles to add to the base triangles
        self.right_push = np.array([[0.1, 0], [0.1, 0], [0.1, 0]])
        self.down_push = np.array([[0, 0.1], [0, 0.1], [0, 0.1]])
        # Create colours to interpolate between
        self.max_colour = np.array([0, 255, 255]) # technically not needed
        self.min_colour = np.array([153, 0, 0])
        # For interpolation, will be multiplying a colour difference with a number [0,1]
        self.colour_difference = np.array([-153, 255, 255])

    def return_offset_triangles(self, x_offset, y_offset):
        yield ((self.right_triangle + self.right_push * x_offset + self.down_push * y_offset) * self.magnification).astype(int)
        yield ((self.left_triangle + self.right_push * x_offset + self.down_push * y_offset) * self.magnification).astype(int)
        yield ((self.top_triangle + self.right_push * x_offset + self.down_push * y_offset) * self.magnification).astype(int)
        yield ((self.bottom_triangle + self.right_push * x_offset + self.down_push * y_offset) * self.magnification).astype(int)

    def return_interpolated_colour(self, factor):
        return self.min_colour + self.colour_difference * factor

    def draw(self, colour_interpolations: list, show_goal=False):
        # Colour interpolations are given in the order of actions: right left up down
        colour_interpolations = iter(colour_interpolations)
        for y_coord in range(10):
            for x_coord in range(10):
                interpolation = next(colour_interpolations)
                # count = 0
                for colour_factor, triangle in zip(interpolation, self.return_offset_triangles(x_coord, y_coord)):
                    # print(count)
                    # count += 1
                    # colour = self.return_interpolated_colour(colour_factor)
                    # print(colour)
                    cv2.fillConvexPoly(self.image, triangle, self.return_interpolated_colour(colour_factor)) # draw triangles
                    polylinepts = np.array([triangle[0], triangle[-1], triangle[1], triangle[-1]]) # start and end points of the lines, outer point to inner point
                    # print(polylinepts)
                    cv2.polylines(self.image, [polylinepts], True, (0, 0, 0), 3)  # draw triangle borders, polylines expects a list that contains a numpy array with coords
                    # cv2.polylines(self.image, triangle.reshape(-1,1,2), True, (0, 0, 0), 3) # draw triangle borders
                    # cv2.polylines(self.image, triangle,
                    #               self.return_interpolated_colour(colour_factor))  # draw triangle borders

        # Show goal state
        if show_goal:
            goal_centre = (int(self.goal_state[0] * self.magnification), int((1 - self.goal_state[1]) * self.magnification))
            goal_radius = int(0.02 * self.magnification)
            goal_colour = (0, 255, 0)
            cv2.circle(self.image, goal_centre, goal_radius, goal_colour, cv2.FILLED)

        # Draw grid lines
        for x_coord in np.arange(0.1, 1, 0.1):
            startpoint = (int(x_coord * self.magnification), 0)
            endpoint = (int(x_coord * self.magnification), int(self.height * self.magnification))
            print(startpoint)
            print(endpoint)
            cv2.line(self.image, startpoint, endpoint, (255, 255, 255), 2)
        for y_coord in np.arange(0.1, 1, 0.1):
            startpoint = (0, int(y_coord * self.magnification))
            endpoint = (int(self.width * self.magnification), int(y_coord * self.magnification))
            cv2.line(self.image, startpoint, endpoint, (255, 255, 255), 2)

        # Show the image
        cv2.imshow("Environment", self.image)
        # This line is necessary to give time for the image to be rendered on the screen
        cv2.waitKey(1)

if __name__ == "__main__":
    qv = QVisualisation(True, 1000)
    # triangle = qv.offset_triangles(1, 1)

    # print(qv.return_colour_scheme(0.5))
    # qv.plot_single_triangle((0, 255, 255))
    # cv2.fillConvexPoly(qv.image, qv.right_triangle, (0, 255, 0))
    # Show the image
    cv2.imshow("Environment", qv.image)
    # This line is necessary to give time for the image to be rendered on the screen
    cv2.waitKey(1)

    # time.sleep(5)



#
# num_coordinates_axis = int(1 / 0.05)
#
# magnification = 100
#
# # image = np.zeros((num_coordinates_axis * magnification, num_coordinates_axis * magnification, 3), dtype=np.uint8)
# image = np.zeros((100, 100, 3), dtype=np.uint8)
# image.fill(0)
# triangle = np.array([[1,100],[100,100],[50,50]])
# triangle *= magnification
# cv2.fillConvexPoly(image, triangle, (150,150,150))
#
#
# cv2.imshow("Environment", image)
# cv2.waitKey(0)
# time.sleep(5)
# #
# # import numpy as np
# # import cv2 as cv
# # # Create a black image
# #
# # height = 100
# # width = 100
# #
# # import numpy as np
# # blank_image = np.zeros((height,width,3), np.uint8)
# #
# # blank_image[:,0:width//2] = (255,0,0)      # (B, G, R)
# # blank_image[:,width//2:width] = (0,255,0)
# #
# # cv.imshow("Environment", blank_image)
# # cv.waitKey(1)
# # time.sleep(5)