import numpy as np
from cell import Cell

class CellEnvironment:
    """
    Simulation environment for cells and vector field.
    Allows optional initialization with predefined positions.
    """
    def __init__(self, num_cells, width, height, grid_size, cell_radius=5, positions=None):
        self.num_cells = num_cells
        self.width = width
        self.height = height
        self.grid_size = grid_size
        self.cell_radius = cell_radius
        self.time_step = 0

        # Initialize cells
        if positions is not None:
            assert len(positions) == num_cells, "Number of positions must match num_cells"
            self.cells = [
                Cell(
                    position=np.array(pos),
                    velocity=np.random.uniform(low=-0.5, high=0.5, size=(2,)),
                    cell_type=np.random.randint(0, 1),
                    grid_size=(width, height),
                    radius=cell_radius
                )
                for pos in positions
            ]
        else:
            self.cells = [
                Cell(
                    position=np.random.uniform(low=cell_radius, high=[width - cell_radius, height - cell_radius]),
                    velocity=np.random.uniform(low=-0.5, high=0.5, size=(2,)),
                    cell_type=np.random.randint(0, 1),
                    grid_size=(width, height),
                    radius=cell_radius
                )
                for _ in range(num_cells)
            ]

        # Vector field and logging
        self.vector_field = np.random.uniform(low=-1, high=1, size=grid_size + (2,))
        self.avg_pairwise_distances = []

    def update(self, dt=1.0):
        for cell in self.cells:
            cell.update_position(self.vector_field, self.cells, dt)
        self.compute_metrics()
        self.time_step += 1

    def compute_metrics(self):
        positions = np.array([cell.position for cell in self.cells])
        n = len(positions)
        if n < 2:
            self.avg_pairwise_distances.append(0.0)
        else:
            dists = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
            avg_dist = dists[np.triu_indices(n, k=1)].mean()
            self.avg_pairwise_distances.append(avg_dist)

    def get_state_log(self):
        return {
            "positions": [cell.position.tolist() for cell in self.cells],
            "avg_pairwise_distance": self.avg_pairwise_distances[-1] if self.avg_pairwise_distances else None,
        }
        
def create_clustering_vector_field(grid_size, center=None):
    """
    Create a vector field that causes cells to cluster at a central point.

    Args:
        grid_size (tuple): Dimensions of the grid (width, height).
        center (tuple): Central point for clustering (default: center of grid).

    Returns:
        np.ndarray: Vector field of shape (grid_width, grid_height, 2).
    """
    grid_width, grid_height = grid_size

    # Default to the center of the grid
    if center is None:
        center = (grid_width / 2, grid_height / 2)

    center_x, center_y = center

    # Create meshgrid for grid coordinates
    x_coords, y_coords = np.meshgrid(np.arange(grid_width), np.arange(grid_height), indexing='ij')

    # Compute direction vectors toward the center
    directions = np.stack([center_x - x_coords, center_y - y_coords], axis=-1)
    magnitudes = np.linalg.norm(directions, axis=-1, keepdims=True)

    # Normalize vectors, avoiding division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        normalized_vectors = np.divide(directions, magnitudes, where=magnitudes > 0)

    # Return the normalized vector field
    return normalized_vectors

def create_scattering_vector_field(grid_size, center=None):
    """
    Create a vector field that causes cells to scatter outward from a central point.

    Args:
        grid_size (tuple): Dimensions of the grid (width, height).
        center (tuple): Central point for scattering (default: center of grid).

    Returns:
        np.ndarray: Vector field of shape (grid_width, grid_height, 2).
    """
    grid_width, grid_height = grid_size

    # Default to the center of the grid
    if center is None:
        center = (grid_width / 2, grid_height / 2)

    center_x, center_y = center

    # Create meshgrid for grid coordinates
    x_coords, y_coords = np.meshgrid(np.arange(grid_width), np.arange(grid_height), indexing='ij')

    # Compute direction vectors away from the center
    directions = np.stack([x_coords - center_x, y_coords - center_y], axis=-1)
    magnitudes = np.linalg.norm(directions, axis=-1, keepdims=True)

    # Normalize vectors, avoiding division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        normalized_vectors = np.divide(directions, magnitudes, where=magnitudes > 0)

    # Return the normalized vector field
    return normalized_vectors

def create_directional_vector_field(grid_size, direction):
    """
    Create a uniform vector field for directional movement (North, East, South, West).

    Args:
        grid_size (tuple): Dimensions of the vector field grid (width, height).
        direction (str): Direction of movement ('north', 'east', 'south', 'west').

    Returns:
        np.ndarray: A vector field of shape (grid_width, grid_height, 2).
    """
    grid_width, grid_height = grid_size

    # Initialize the vector field
    vector_field = np.zeros((grid_width, grid_height, 2))

    # Assign vector values based on direction
    if direction == "north":
        vector_field[..., 1] = -1  # Negative Y for upward movement
    elif direction == "east":
        vector_field[..., 0] = 1   # Positive X for rightward movement
    elif direction == "south":
        vector_field[..., 1] = 1   # Positive Y for downward movement
    elif direction == "west":
        vector_field[..., 0] = -1  # Negative X for leftward movement
    else:
        raise ValueError("Invalid direction. Choose from 'north', 'east', 'south', 'west'.")

    return vector_field
    
def create_random_vector_field(grid_size, value_range=(-1, 1)):
    """
    Create a random vector field with vectors in the specified range.
    
    Parameters:
    grid_size (tuple): A tuple (grid_width, grid_height).
    value_range (tuple): The range of values for each component of the vectors.
    
    Returns:
    np.ndarray: A grid of vectors with shape (grid_height, grid_width, 2).
    """
    low, high = value_range
    grid_width, grid_height = grid_size
    vector_field = np.random.uniform(low, high, (grid_height, grid_width, 2))
    return vector_field

def avg_distance_to_centroid(positions, target=(250, 250)):
    target = np.array(target)
    distances = np.linalg.norm(positions - target, axis=1)
    return np.mean(distances)  # Minimize this score
