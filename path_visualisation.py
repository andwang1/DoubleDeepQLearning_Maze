import cv2
import numpy as np


class PathVisualisation:
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
        # Create colours to interpolate between
        self.end_colour = np.array([0, 255, 0]) # technically not needed
        self.start_colour = np.array([0, 0, 255])
        # For interpolation, will be multiplying a colour difference with a number [0,1]
        self.colour_difference = np.array([0, 255, -255])

    def return_interpolated_colour(self, factor):
        return self.start_colour + self.colour_difference * factor

    def draw(self, line_coordinates: list, show_goal=False):
        # Draw white grid lines
        for x_coord in np.arange(0.1, 1, 0.1):
            startpoint = (int(x_coord * self.magnification), 0)
            endpoint = (int(x_coord * self.magnification), int(self.height * self.magnification))
            # cv line requires the start and endpoints to be given as tuples of ints as positional parameters
            cv2.line(self.image, startpoint, endpoint, (255, 255, 255), 2)
        for y_coord in np.arange(0.1, 1, 0.1):
            startpoint = (0, int(y_coord * self.magnification))
            endpoint = (int(self.width * self.magnification), int(y_coord * self.magnification))
            cv2.line(self.image, startpoint, endpoint, (255, 255, 255), 2)

        print(line_coordinates)
        # for index in range(len(line_coordinates) - 1):
        #     colour = self.return_interpolated_colour(index / (len(line_coordinates) - 2))
        #     if index == 0:
        #         # plot start state circle
        #         pass
        #     start_coordinates = (line_coordinates[index] * self.magnification).astype(int)
        #     start_tuple = tuple([start_coordinates[i] for i in range(len(start_coordinates))])
        #     end_coordinates = (line_coordinates[index + 1] * self.magnification).astype(int)
        #     end_tuple = tuple([end_coordinates[i] for i in range(len(end_coordinates))])
        #     cv2.line(self.image, start_tuple, end_tuple, colour, 2)
        # cv2.line(self.image, (int(0.15 * self.magnification), int((1 - 0.15) * self.magnification)), (int(0.15 * self.magnification), int(0.25 * self.magnification)), (255,0,0), 2)
        line_counter = 0
        for index, start_coordinates in enumerate(line_coordinates[:-1]):
                colour = self.return_interpolated_colour(index / (len(line_coordinates) - 2))
                end_coordinates = line_coordinates[index + 1]
                print(start_coordinates)
                print(end_coordinates)
                start_x, start_y = round(start_coordinates[0], 2), 1 - round(start_coordinates[1], 2)
                end_x, end_y = round(end_coordinates[0], 2), 1 - round(end_coordinates[1], 2)
                # Need to flip y-values as origin is top left corner
                start_tuple = (int(start_x * self.magnification), int(start_y * self.magnification))
                # end_x, end_y = end_coordinates
                # Need to flip y-values as origin is top left corner
                end_tuple = (int(end_x * self.magnification), int(end_y * self.magnification))
                print(start_tuple)
                print(end_tuple)
                line_counter += 1
                print(line_counter)
                # cv line requires the start and endpoints to be given as tuples of ints as positional parameters
                cv2.line(self.image, start_tuple, end_tuple, colour, 3)

        # Show goal state
        if show_goal:
            goal_centre = (int(self.goal_state[0] * self.magnification), int((1 - self.goal_state[1]) * self.magnification))
            goal_radius = int(0.02 * self.magnification)
            goal_colour = (0, 255, 0)
            cv2.circle(self.image, goal_centre, goal_radius, goal_colour, cv2.FILLED)

        # Draw first and last state circles

        # Show the image
        cv2.imshow("Environment", self.image)
        # This line is necessary to give time for the image to be rendered on the screen
        cv2.waitKey(1)
