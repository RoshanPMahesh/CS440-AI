# geometry.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Joshua Levine (joshua45@illinois.edu)
# Inspired by work done by James Gao (jamesjg2@illinois.edu) and Jongdeog Lee (jlee700@illinois.edu)

"""
This file contains geometry functions necessary for solving problems in MP5
"""

import numpy as np
from alien import Alien
from typing import List, Tuple
from copy import deepcopy
import math as mh


def does_alien_touch_wall(alien: Alien, walls: List[Tuple[int]]):
    """Determine whether the alien touches a wall

        Args:
            alien (Alien): Instance of Alien class that will be navigating our map
            walls (list): List of endpoints of line segments that comprise the walls in the maze in the format
                         [(startx, starty, endx, endx), ...]

        Return:
            True if touched, False if not
    """
    # different conditions based on if the alien is a circle or not due to different lengths/radius
    if alien.is_circle() == True:
        # checks the distance from the alien, which is a point, to each wall line segment
        # if the distance is within the circle radius, that means the edge of the alien is touching the wall
        alien_coordinates = alien.get_centroid()
        circle_radius = alien.get_width()
        
        for each_wall in walls:
            startx, starty, endx, endy = each_wall 
            if point_segment_distance(alien_coordinates, ((startx, starty), (endx, endy))) <= circle_radius:
                return True
    else:
        # checks the distance from the alien, which is a line segment, to each wall line segment
        # if the distance is within the width of the line segment, then the alien is touching the wall
        head, tail = alien.get_head_and_tail()
        alien_width = alien.get_width()

        for each_wall in walls:
            startx, starty, endx, endy = each_wall
            if segment_distance((head, tail), ((startx, starty), (endx, endy))) <= alien_width:
                return True

    return False


def is_alien_within_window(alien: Alien, window: Tuple[int]):
    """Determine whether the alien stays within the window

        Args:
            alien (Alien): Alien instance
            window (tuple): (width, height) of the window
    """
    alien_x_coord, alien_y_coord = alien.get_centroid()
    window_width, window_height = window

    # need to check each of the 3 shapes if they are within the window since different orientations, radius, lengths, etc
    if alien.is_circle():
        # need to check if the edges of the circle are within the window, which means checking if the radius of the circle is inbounds
        radius = alien.get_width()
        if ((radius <= alien_x_coord <= window_width - radius) and (radius <= alien_y_coord <= window_height - radius)):
            return True
        else:
            return False
    else:
        # this is for horizontal and vertical alien shapes
        head, tail = alien.get_head_and_tail()
        alien_width = alien.get_width() 

        # bounds checks are slightly different among vertical and horizontal shapes but same idea
        # need to check if the edges of the oblong are within the window, which means checking if the width from the center of the oblong is inbounds
        if alien.get_shape() == 'Vertical':
            if ((alien_width < alien_x_coord < window_width - alien_width) and (tail[1] < window_height - alien_width) and (head[1] > alien_width)):
                return True
            else:
                return False
        else:
            if ((head[0] < window_width - alien_width) and (tail[0] > alien_width) and (alien_width < alien_y_coord < window_height - alien_width)):
                return True
            else:
                return False


def is_point_in_polygon(point, polygon):
    """Determine whether a point is in a parallelogram.
    Note: The vertex of the parallelogram should be clockwise or counter-clockwise.

        Args:
            point (tuple): shape of (2, ). The coordinate (x, y) of the query point.
            polygon (tuple): shape of (4, 2). The coordinate (x, y) of 4 vertices of the parallelogram.
    """
    x_coord, y_coord = point
    poly_tuple_len = len(polygon)
    less_len = poly_tuple_len - 1
    poly_check = False
    
    # iterates throughout the polygon
    for i in range(poly_tuple_len):
        poly_x, poly_y = polygon[i]
        other_x, other_y = polygon[less_len]
        next_x, next_y = polygon[(i + 1) % poly_tuple_len] 
        
        # gets all of the line segments
        y_min = min(poly_y, next_y)
        y_max = max(poly_y, next_y)
        x_min = min(poly_x, next_x)
        x_max = max(poly_x, next_x)

        # next 3 if statements check to see if the point is on the edge of the polygon
        if (poly_x == next_x) and (poly_x == x_coord) and (y_min < y_coord < y_max):
            return True
        
        if (poly_y == next_y) and (poly_y == y_coord) and (x_min < x_coord < x_max):
            return True
        
        # checks to see if the point is on a line of the polygon rather than a specific point on the polygon
        if (poly_y != next_y) and (poly_x != next_x):
            slope = (next_y - poly_y)/(next_x - poly_x)
            y_inter = poly_y - (slope * poly_x)

            # y = mx + b formula check
            if (y_coord == (slope*x_coord) + y_inter) and (x_min < x_coord < x_max):
                return True
        
        # checks to see if the point is inside of the polygon
        if (poly_y < y_coord and other_y >= y_coord) or (other_y < y_coord and poly_y >= y_coord):
            long_math = poly_x + (y_coord - poly_y)/(other_y - poly_y) * (other_x - poly_x)
            if long_math < x_coord:
                # flips poly_check
                if poly_check == True:
                    poly_check = False
                elif poly_check == False:
                    poly_check = True

        less_len = i
    
    return poly_check


def does_alien_path_touch_wall(alien: Alien, walls: List[Tuple[int]], waypoint: Tuple[int, int]):
    """Determine whether the alien's straight-line path from its current position to the waypoint touches a wall

        Args:
            alien (Alien): the current alien instance
            walls (List of tuple): List of endpoints of line segments that comprise the walls in the maze in the format
                         [(startx, starty, endx, endx), ...]
            waypoint (tuple): the coordinate of the waypoint where the alien wants to move

        Return:
            True if touched, False if not
    """
    alien_position = alien.get_centroid()
    radius = alien.get_width()

    # iterates through all the walls and checks the alien path for each shape
    # have to check if the alien is already at the waypoint or not since being at the waypoint is equal to just a point, but not at the waypoint is a line segment
    for each_wall in walls:
        startx, starty, endx, endy = each_wall
        if alien.is_circle():
            # if the alien is a circle and at the waypoint, then need to check the distance between the current position and the walls
            # if this distance is less than or equal to the radius, then the alien path touches the wall since it touches the outside of the alien
            if (alien_position == waypoint):
                if point_segment_distance(alien_position, ((startx, starty), (endx, endy))) <= radius:
                    return True
            else:
                # checks to see if the distance between the 2 line segments is less than or equal to the radius or else the alien can't fit
                if segment_distance((alien_position, waypoint), ((startx, starty), (endx, endy))) <= radius:
                    return True
        else:
            alien_line_seg = alien.get_head_and_tail()

            # same thing as the circle alien but with a line segment to represent the vertical or horizontal alien
            if (alien_position == waypoint):
                if segment_distance(alien_line_seg, ((startx, starty), (endx, endy))) <= radius:
                    return True
            else:
                alien_half_len = alien.get_length() / 2

                # if the alien is horizontal, then changes the x coordinate to be the waypoint +- half the alien length due to the alien shape
                # if the alien is vertical, then changes the y coordinate to be the waypoint +- half the alien length due to the alien shape
                if alien.get_shape() == 'Horizontal':
                    heads = (waypoint[0] - alien_half_len, waypoint[1])
                    tails = (waypoint[0] + alien_half_len, waypoint[1])
                else:
                    heads = (waypoint[0], waypoint[1] - alien_half_len)
                    tails = (waypoint[0], waypoint[1] + alien_half_len)

                # checks if any of these line segments are less than or equal to the radius since then the alien can't fit
                if segment_distance((alien_line_seg[0], heads), ((startx, starty), (endx, endy))) <= radius:
                    return True
                
                if segment_distance((alien_line_seg[1], tails), ((startx, starty), (endx, endy))) <= radius:
                    return True
                
                if segment_distance((alien_position, waypoint), ((startx, starty), (endx, endy))) <= radius:
                    return True

    return False


def point_segment_distance(p, s):
    """Compute the distance from the point to the line segment.

        Args:
            p: A tuple (x, y) of the coordinates of the point.
            s: A tuple ((x1, y1), (x2, y2)) of coordinates indicating the endpoints of the segment.

        Return:
            Euclidean distance from the point to the line segment.
    """
    # learned about the use of projections from: https://math.stackexchange.com/questions/62633/orthogonal-projection-of-a-point-onto-a-line
    x, y = p
    x1, y1 = s[0]
    x2, y2 = s[1]
    line_seg_dist = mh.sqrt(((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1)))

    # projection of the point onto the line segment
    projection = (((x - x1) * (x2 - x1)) + ((y - y1) * (y2 - y1))) / (line_seg_dist * line_seg_dist)

    # uses projection to get the point on the line segment that is closest to the original point
    new_x = (projection * (x2 - x1)) + x1
    new_y = (projection * (y2 - y1)) + y1 

    # if the projection is 0 or 1, then the distance is just from the point to the endpoint of the line segment
    # if the projection is between 0 and 1, then the distance is the point from the projection to the original point
    if projection <= 0:
        return mh.sqrt(((x - x1) * (x - x1)) + ((y - y1) * (y - y1)))
    elif projection >= 1:
        return mh.sqrt(((x - x2) * (x - x2)) + ((y - y2) * (y - y2)))
    else:
        return mh.sqrt(((x - new_x) * (x - new_x)) + ((y - new_y) * (y - new_y)))


def direction(a, b, c):
    dir = ((b[1] - a[1]) * (c[0] - b[0])) - ((b[0] - a[0]) * (c[1] - b[1]))
    if dir < 0:
        # counter-clockwise direction
        return -1
    elif dir > 0:
        # clockwise direction
        return 1
    else:
        # collinear
        return 0
    

def check_collinear_collision(a, b, c):
    # if the (x, y) coordinate is in range of the (x, y) coordinates of the line segment, then it's a collinear collision
    if (a[0] <= max(b[0], c[0]) and a[0] >= min(b[0], c[0]) and a[1] <= max(b[1], c[1]) and a[1] >= min(b[1], c[1])):
        return True
    else:
        return False


def do_segments_intersect(s1, s2):
    """Determine whether segment1 intersects segment2.

        Args:
            s1: A tuple of coordinates indicating the endpoints of segment1.
            s2: A tuple of coordinates indicating the endpoints of segment2.

        Return:
            True if line segments intersect, False if not.
    """
    # learned about the direction of ordered triplet of points from: https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
    s1_end1, s1_end2 = s1
    s2_end1, s2_end2 = s2

    # computes the direction of the triplet of points: counter-clockwise, clockwise, or collinear
    dir1 = direction(s1_end1, s1_end2, s2_end1)
    dir2 = direction(s1_end1, s1_end2, s2_end2)
    dir3 = direction(s2_end1, s2_end2, s1_end1)
    dir4 = direction(s2_end1, s2_end2, s1_end2)

    # this condition means that the segments intersect
    if dir1 != dir2 and dir3 != dir4:
        return True

    # all the collinear cases where the segments still intersect
    if dir1 == 0 and check_collinear_collision(s1_end1, s2_end1, s2_end2) == True:
        return True
    elif dir2 == 0 and check_collinear_collision(s1_end2, s2_end1, s2_end2) == True:
        return True
    elif dir3 == 0 and check_collinear_collision(s2_end1, s1_end1, s1_end2) == True:
        return True
    elif dir4 == 0 and check_collinear_collision(s2_end2, s1_end1, s1_end2) == True:
        return True
    else:
        return False


def segment_distance(s1, s2):
    """Compute the distance from segment1 to segment2.  You will need `do_segments_intersect`.

        Args:
            s1: A tuple of coordinates indicating the endpoints of segment1.
            s2: A tuple of coordinates indicating the endpoints of segment2.

        Return:
            Euclidean distance between the two line segments.
    """
    if do_segments_intersect(s1, s2) == True:
        # the segments intersect so the distance is 0
        return 0
    else:
        s1_end1, s1_end2 = s1
        s2_end1, s2_end2 = s2

        # gets the distance from each endpoint of each line segment to the other line segment
        distance1 = point_segment_distance(s1_end1, s2)
        distance2 = point_segment_distance(s1_end2, s2)
        distance3 = point_segment_distance(s2_end1, s1)
        distance4 = point_segment_distance(s2_end2, s1)

        # minimum calculated distance is the euclidean distance
        return min(distance1, distance2, distance3, distance4)
    


if __name__ == '__main__':

    from geometry_test_data import walls, goals, window, alien_positions, alien_ball_truths, alien_horz_truths, \
        alien_vert_truths, point_segment_distance_result, segment_distance_result, is_intersect_result, waypoints


    # Here we first test your basic geometry implementation
    def test_point_segment_distance(points, segments, results):
        num_points = len(points)
        num_segments = len(segments)
        for i in range(num_points):
            p = points[i]
            for j in range(num_segments):
                seg = ((segments[j][0], segments[j][1]), (segments[j][2], segments[j][3]))
                cur_dist = point_segment_distance(p, seg)
                assert abs(cur_dist - results[i][j]) <= 10 ** -3, \
                    f'Expected distance between {points[i]} and segment {segments[j]} is {results[i][j]}, ' \
                    f'but get {cur_dist}'


    def test_do_segments_intersect(center: List[Tuple[int]], segments: List[Tuple[int]],
                                   result: List[List[List[bool]]]):
        for i in range(len(center)):
            for j, s in enumerate([(40, 0), (0, 40), (100, 0), (0, 100), (0, 120), (120, 0)]):
                for k in range(len(segments)):
                    cx, cy = center[i]
                    st = (cx + s[0], cy + s[1])
                    ed = (cx - s[0], cy - s[1])
                    a = (st, ed)
                    b = ((segments[k][0], segments[k][1]), (segments[k][2], segments[k][3]))
                    if do_segments_intersect(a, b) != result[i][j][k]:
                        if result[i][j][k]:
                            assert False, f'Intersection Expected between {a} and {b}.'
                        if not result[i][j][k]:
                            assert False, f'Intersection not expected between {a} and {b}.'


    def test_segment_distance(center: List[Tuple[int]], segments: List[Tuple[int]], result: List[List[float]]):
        for i in range(len(center)):
            for j, s in enumerate([(40, 0), (0, 40), (100, 0), (0, 100), (0, 120), (120, 0)]):
                for k in range(len(segments)):
                    cx, cy = center[i]
                    st = (cx + s[0], cy + s[1])
                    ed = (cx - s[0], cy - s[1])
                    a = (st, ed)
                    b = ((segments[k][0], segments[k][1]), (segments[k][2], segments[k][3]))
                    distance = segment_distance(a, b)
                    assert abs(result[i][j][k] - distance) <= 10 ** -3, f'The distance between segment {a} and ' \
                                                                        f'{b} is expected to be {result[i]}, but your' \
                                                                        f'result is {distance}'


    def test_helper(alien: Alien, position, truths):
        alien.set_alien_pos(position)
        config = alien.get_config()

        touch_wall_result = does_alien_touch_wall(alien, walls)
        in_window_result = is_alien_within_window(alien, window)

        assert touch_wall_result == truths[
            0], f'does_alien_touch_wall(alien, walls) with alien config {config} returns {touch_wall_result}, ' \
                f'expected: {truths[0]}'
        assert in_window_result == truths[
            2], f'is_alien_within_window(alien, window) with alien config {config} returns {in_window_result}, ' \
                f'expected: {truths[2]}'


    def test_check_path(alien: Alien, position, truths, waypoints):
        alien.set_alien_pos(position)
        config = alien.get_config()

        for i, waypoint in enumerate(waypoints):
            path_touch_wall_result = does_alien_path_touch_wall(alien, walls, waypoint)

            assert path_touch_wall_result == truths[
                i], f'does_alien_path_touch_wall(alien, walls, waypoint) with alien config {config} ' \
                    f'and waypoint {waypoint} returns {path_touch_wall_result}, ' \
                    f'expected: {truths[i]}'

            # Initialize Aliens and perform simple sanity check.


    alien_ball = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Ball', window)
    test_helper(alien_ball, alien_ball.get_centroid(), (False, False, True))

    alien_horz = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Horizontal', window)
    test_helper(alien_horz, alien_horz.get_centroid(), (False, False, True))

    alien_vert = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Vertical', window)
    test_helper(alien_vert, alien_vert.get_centroid(), (True, False, True))

    edge_horz_alien = Alien((50, 100), [100, 0, 100], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Horizontal',
                            window)
    edge_vert_alien = Alien((200, 70), [120, 0, 120], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Vertical',
                            window)

    # Test validity of straight line paths between an alien and a waypoint
    test_check_path(alien_ball, (30, 120), (False, True, True), waypoints)
    test_check_path(alien_horz, (30, 120), (False, True, False), waypoints)
    test_check_path(alien_vert, (30, 120), (True, True, True), waypoints)

    centers = alien_positions
    segments = walls
    test_point_segment_distance(centers, segments, point_segment_distance_result)
    test_do_segments_intersect(centers, segments, is_intersect_result)
    test_segment_distance(centers, segments, segment_distance_result)

    for i in range(len(alien_positions)):
        test_helper(alien_ball, alien_positions[i], alien_ball_truths[i])
        test_helper(alien_horz, alien_positions[i], alien_horz_truths[i])
        test_helper(alien_vert, alien_positions[i], alien_vert_truths[i])

    # Edge case coincide line endpoints
    test_helper(edge_horz_alien, edge_horz_alien.get_centroid(), (True, False, False))
    test_helper(edge_horz_alien, (110, 55), (True, True, True))
    test_helper(edge_vert_alien, edge_vert_alien.get_centroid(), (True, False, True))

    print("Geometry tests passed\n")
