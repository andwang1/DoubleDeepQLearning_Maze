import cv2
import numpy as np
import time
#

class QVisualisation:
    def __init__(self, display, magnification):
        # Set whether the environment should be displayed after every step
        self.display = display
        # Set the magnification factor of the display
        self.magnification = magnification
        # Set the initial state of the agent
        self.init_state = np.array([0.15, 0.15], dtype=np.float32)
        # Set the initial state of the goal
        self.goal_state = np.array([0.75, 0.85], dtype=np.float32)
        # Set the space which the obstacle occupies
        self.obstacle_space = np.array([[0.3, 0.5], [0.3, 0.6]], dtype=np.float32)
        # Set the width and height of the environment
        self.width = 1.0
        self.height = 1.0
        # Create an image which will be used to display the environment
        self.image = np.zeros([int(self.magnification * self.height), int(self.magnification * self.width), 3], dtype=np.uint8)

        # Base triangles to iterate over
        self.top_triangle = np.array([[0, 0], [0.1, 0], [0.05, 0.05]]) * self.magnification
        self.bottom_triangle = np.array([[0.1, 0.1], [0, 0.1], [0.05, 0.05]]) * self.magnification
        self.right_triangle = np.array([[0.1, 0.1], [0.1, 0], [0.05, 0.05]]) * self.magnification
        self.left_triangle = np.array([[0, 0], [0, 0.1], [0.05, 0.05]]) * self.magnification

        # Offset triangles to add
        self.right_push = np.array([[0.1,0],[0.1,0],[0.1,0]]) * self.magnification
        self.down_push = np.array([[0,0.1],[0,0.1],[0,0.1]]) * self.magnification

        # Colours
        self.max_colour = np.array([0, 255, 255]) # technically not needed
        self.min_colour = np.array([153, 0, 0])
        self.colour_difference = np.array([-153, 255, 255]) # will be doing self.min_colour + self.colour_difference * colour interpolations to get the colours for each triangle


    def offset_triangles(self, x_offset, y_offset):
        # print(self.right_triangle + self.right_push)
        # print(self.right_triangle + self.right_push* x_offset)
        # print(self.right_push * x_offset)
        # print(self.down_push * y_offset)
        # print(self.right_triangle + self.right_push * x_offset + self.down_push * y_offset)
        yield (self.right_triangle + self.right_push * x_offset + self.down_push * y_offset).astype(int)
        yield (self.left_triangle + self.right_push * x_offset + self.down_push * y_offset).astype(int)
        yield (self.top_triangle + self.right_push * x_offset + self.down_push * y_offset).astype(int)
        yield (self.bottom_triangle + self.right_push * x_offset + self.down_push * y_offset).astype(int)

    # loop through every state (0,1 steps to 1)
    #

    def return_colour_scheme(self, colour_factor):
        return tuple((self.min_colour + self.colour_difference * colour_factor))

    #
    # def populate_triangles(self, colour_interpolations):
    #     # loop through interpolations
    #     # order is right left up down
    #     for y_coord in range(10):
    #         for x_coord in range(10):
    #             for interpolation, triangle in zip(colour_interpolations, self.offset_triangles(x_coord, y_coord)):
    #                 cv2.fillConvexPoly(self.image, triangle, interpolation)
    #

    def draw(self, colour_interpolations):
        # Create a black image
        self.image.fill(0)

        # loop through interpolations
        # order is right left up down
        for y_coord in range(10):
            for x_coord in range(10):
                for interpolation in colour_interpolations:
                    count = 0
                    for colour_factor, triangle in zip(interpolation, self.offset_triangles(x_coord, y_coord)):
                        print(count)
                        count += 1
                        colour = self.return_colour_scheme(colour_factor)
                        print(colour)
                        cv2.fillConvexPoly(self.image, triangle, colour)
        # Show the image
        cv2.imshow("Environment", self.image)
        # This line is necessary to give time for the image to be rendered on the screen
        cv2.waitKey(1)


qv = QVisualisation(True, 1000)
# triangle = qv.offset_triangles(1, 1)
# print(next(triangle))
print(qv.return_colour_scheme(0.5))
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