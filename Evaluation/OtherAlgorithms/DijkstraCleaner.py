import heapq
import sys
sys.path.append('.')
import numpy as np
import matplotlib.pyplot as plt

def dijkstra(graph, start):
    # Initialize distances and priority queue
    distances = {vertex: float('infinity') for vertex in graph}
    distances[start] = 0
    priority_queue = [(0, start)]
    predecessors = {vertex: None for vertex in graph}
    
    while priority_queue:
        current_distance, current_vertex = heapq.heappop(priority_queue)

        if current_distance > distances[current_vertex]:
            continue

        for neighbor, weight in graph[current_vertex].items():
            distance = current_distance + weight

            if distance < distances[neighbor]:
                distances[neighbor] = distance
                predecessors[neighbor] = current_vertex
                heapq.heappush(priority_queue, (distance, neighbor))
                
    return distances, predecessors
def isnot_reachable(grid, current_position, next_position):
		""" Check if the next position is reachable or navigable """
		if grid[int(next_position[0]), int(next_position[1])] == 0:
			return True 
		x, y = next_position
		dx = x - current_position[0]
		dy = y - current_position[1]
		steps = max(abs(dx), abs(dy))
		dx = dx / steps if steps != 0 else 0
		dy = dy / steps if steps != 0 else 0
		reachable_positions = True
		for step in range(1, steps + 1):
			px = round(current_position[0] + dx * step)
			py = round(current_position[1] + dy * step)
			if grid[px, py] != 1:
				reachable_positions = False
				break

		return not reachable_positions
def get_path(predecessors, start, end):
    path = []
    while end is not None:
        path.append(end)
        end = predecessors[end]
    path.reverse()
    return path

def closest_destination(graph, source, destinations):
    distances, predecessors = dijkstra(graph, source)
    closest_vertex = None
    min_distance = float('infinity')
    
    for destination in destinations:
        if distances[destination] < min_distance:
            min_distance = distances[destination]
            closest_vertex = destination
            
    if closest_vertex is not None:
        path = get_path(predecessors, source, closest_vertex)
        return closest_vertex, min_distance, path
    else:
        return None, float('infinity'), []

def grid_to_graph(grid,directions):
    rows = grid.shape[0]
    cols = grid.shape[1]
    graph = {}

    # Directions for 8 adjacent cells (including diagonals)
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    for x in range(rows):
        for y in range(cols):
            if grid[x,y] == 1:  # Assuming 1 represents a navigable cell
                graph[(x, y)] = {}
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < rows and 0 <= ny < cols and grid[nx,ny] == 1:
                        if isnot_reachable(grid, (x, y), (nx, ny)):
                            continue
                        graph[(x, y)][(nx, ny)] = np.linalg.norm(np.array([x,y]) - np.array([nx,ny]))  # Assuming all edges have a weight of 1

    return graph

def visit_all_destinations(graph, source, destinations,sc_map):
    current_position = source
    visited = []
    len_dest = len(destinations)
    while len(visited) < len_dest:
        next_destination, min_distance, path = closest_destination(graph, current_position, destinations)
        if next_destination is None:
            break
        # Mark destination as visited
        visited.append(next_destination)
        # Update destinations
        destinations.remove(next_destination)
        plot_grid(sc_map, current_position, next_destination, destinations, path)
        current_position = next_destination
    return path

def plot_grid(grid, source, closest, destinations, path):
    rows = len(grid)
    cols = len(grid[0])
    image = np.zeros((rows, cols, 3), dtype=np.uint8)

    for x in range(rows):
        for y in range(cols):
            if grid[x][y] == 1:
                image[x, y] = [255, 255, 255]  # White for navigable
            else:
                image[x, y] = [0, 0, 0]  # Black for non-navigable

    # Mark path
    for px, py in path:
        image[px, py] = [0, 0, 255]  # Blue for path
        
    # Mark source
    sx, sy = source
    image[sx, sy] = [255, 255, 0]  # Yellow for source

    # Mark closest destination
    if closest:
        cx, cy = closest
        image[cx, cy] = [0, 255, 0]  # Green for closest destination

    # Mark other destinations
    for destination in destinations:
        if destination != closest:
            dx, dy = destination
            image[dx, dy] = [255, 0, 0]  # Red for other destinations
            
    plt.imshow(image)
    plt.show()
if __name__ == '__main__':
    sc_map = np.genfromtxt('Environment/Maps/malaga_port.csv', delimiter=',')

    N = 4
    initial_positions = np.array([[12, 7], [14, 5], [16, 3], [18, 1]])[:N, :]
    visitable = np.column_stack(np.where(sc_map == 1))
    # Example grid (1 represents a navigable cell, 0 represents a non-navigable cell)
    grid = [
        [1, 1, 1, 0],
        [1, 1, 1, 1],
        [0, 1, 1, 1],
        [1, 0, 1, 1]
    ]

    move_length = 2
    number_of_actions = 8
    angle_set = np.linspace(0, 2 * np.pi, number_of_actions, endpoint=False)
    movement = [np.round(np.array([move_length * np.cos(angle), move_length * np.sin(angle)])).astype(int) for angle in angle_set]
    graph = grid_to_graph(sc_map, movement)
    """
    for vertex, edges in graph.items():
        print(f"Vertex {vertex}: {edges}")
"""
    # Source and destinations
    source = (20, 0)
    destinations = [(24, 4),(53, 20), (0, 23)]

    #closest_vertex, min_distance, path = closest_destination(graph, source, destinations)
    path = visit_all_destinations(graph, source, destinations,sc_map)
    # plot_grid(sc_map, source, destinations, path)
    #print(f"Closest destination: {closest_vertex} with distance {min_distance}")
    #plot_grid(sc_map.tolist(), source, closest_vertex, destinations, path)
