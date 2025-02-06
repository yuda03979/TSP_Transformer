import itertools
import math
import random
import numpy as np
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import networkx as nx
from scipy.spatial import distance_matrix


class Solver:

    def __init__(self):
        self.num_workers = 5

    @staticmethod
    def brute_force(board: list) -> tuple:
        """
        Solve TSP using a brute-force approach (O(n!)).
        :param board: List of (i, j) points.
        :return: Tuple (Ordered path as a list of (i, j), total path length)
        """
        min_path = None
        min_length = float("inf")

        def euclidean_distance(p1, p2):
            return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

        # since the solution will be more like the order of board
        random.shuffle(board)

        for perm in itertools.permutations(board):
            total_length = sum(euclidean_distance(perm[i], perm[i + 1]) for i in range(len(perm) - 1))

            if total_length < min_length:
                min_length = total_length
                min_path = perm

        return list(min_path), min_length

    @staticmethod
    def greedy_tsp(points):
        """
        Solve TSP using a greedy nearest neighbor approach, starting from each city
        and returning the shortest possible route.

        :param points: List of (x, y) coordinates.
        :return: Tuple (Best ordered path as a list of (x, y), total shortest path length)
        """
        points = np.array(points)
        num_points = len(points)

        if num_points <= 1:
            return points.tolist(), 0  # No need to solve TSP for 1 point

        def euclidean_distance(p1, p2):
            return np.linalg.norm(points[p1] - points[p2])

        def greedy_from(start):
            """Greedy TSP starting from a given city index."""
            visited = set()
            path = [start]
            visited.add(start)
            total_distance = 0

            for _ in range(num_points - 1):
                last = path[-1]
                nearest, min_dist = None, float("inf")

                for i in range(num_points):
                    if i not in visited:
                        dist = euclidean_distance(last, i)
                        if dist < min_dist:
                            nearest, min_dist = i, dist

                if nearest is not None:
                    path.append(nearest)
                    visited.add(nearest)
                    total_distance += min_dist

            # Convert indices to (x, y) tuples
            return [tuple(points[i]) for i in path], total_distance

        # Try starting from each city and find the best route
        best_route, best_distance = None, float("inf")
        for start in range(num_points):
            route, distance = greedy_from(start)
            if distance < best_distance:
                best_route, best_distance = route, distance

        return best_route, best_distance

    @staticmethod
    def solve(board):
        if len(board) <= 15:
            return Solver.brute_force(board)
        elif len(board) <= 30:
            return Solver.simulated_annealing_tsp(board)
        else:
            return Solver.greedy_tsp(board)

