import numpy as np

class Cell:
    """
    Represents an individual cell in the simulation.
    """
    def __init__(self, position, velocity, cell_type, grid_size, radius=5):
        """
        Initialize a cell.
        
        Args:
            position (np.ndarray): Initial position of the cell.
            velocity (np.ndarray): Initial velocity of the cell.
            cell_type (int): Type of the cell (used for interaction rules).
            grid_size (tuple): Dimensions of the grid.
            radius (float): Radius of the cell for interactions and visualization.
        """
        self.position = np.array(position, dtype=float)
        self.trajectory = [self.position.copy()]  # ⬅️ Save initial position
        self.velocity = np.array(velocity, dtype=float)
        self.cell_type = cell_type
        self.grid_size = grid_size
        self.radius = radius
        self.sense_range = 5 * self.radius

    def update_position(self, vector_field, other_cells, dt=1.0):
        """
        Update the cell's position based on the vector field and local interactions,
        stopping movement when cells collide with the environment edges.
        
        Args:
            vector_field (np.ndarray): The external vector field influencing the cell.
        other_cells (list[Cell]): List of all other cells in the environment.
        dt (float): Time step for the simulation.
        """
        
        x_frac, x_int = np.modf(self.position[0] / self.grid_size[0] * vector_field.shape[0])
        y_frac, y_int = np.modf(self.position[1] / self.grid_size[1] * vector_field.shape[1])
        x_int, y_int = int(x_int), int(y_int)
        x_int = min(max(x_int, 0), vector_field.shape[0] - 1)
        y_int = min(max(y_int, 0), vector_field.shape[1] - 1)
        
        v00 = vector_field[x_int, y_int]
        v10 = vector_field[min(x_int + 1, vector_field.shape[0] - 1), y_int]
        v01 = vector_field[x_int, min(y_int + 1, vector_field.shape[1] - 1)]
        v11 = vector_field[min(x_int + 1, vector_field.shape[0] - 1), min(y_int + 1, vector_field.shape[1] - 1)]

        vx, vy = (
            (1 - x_frac) * (1 - y_frac) * v00 +
            x_frac * (1 - y_frac) * v10 +
            (1 - x_frac) * y_frac * v01 +
            x_frac * y_frac * v11
        )
        vector_influence = np.array([vx, vy])
        
        # Combine field influence with current velocity
        self.velocity += vector_influence
        
        # Cap the velocity for stability
        max_speed = np.max(np.linalg.norm(vector_field, axis=-1))
        speed = np.linalg.norm(self.velocity)
        if speed > max_speed:
            self.velocity *= max_speed / speed
        
        # Update position temporarily
        new_position = self.position + self.velocity * dt
        self.trajectory.append(self.position.copy())  # ⬅️ Save position every step
        
        boundary_repulsion = 10
        
        # Check and enforce boundary conditions
        if new_position[0] <= self.radius:
            new_position[0] = self.radius
            #self.velocity[0] = 0  # Stop horizontal movement
            self.velocity[0] = -self.velocity[0]  # Reverse horizontal velocity
            self.velocity[0] += boundary_repulsion * (self.radius - new_position[0]) / self.radius
        elif new_position[0] >= self.grid_size[0] - self.radius:
            new_position[0] = self.grid_size[0] - self.radius
            #self.velocity[0] = 0  # Stop horizontal movement
            self.velocity[0] = -self.velocity[0]  # Reverse horizontal velocity
            self.velocity[0] -= boundary_repulsion * (new_position[0] - (self.grid_size[0] - self.radius)) / self.radius
        
        if new_position[1] <= self.radius:
            new_position[1] = self.radius
            #self.velocity[1] = 0  # Stop vertical movement
            self.velocity[1] = -self.velocity[1]  # Reverse vertical velocity
            self.velocity[1] += boundary_repulsion * (self.radius - new_position[1]) / self.radius
        elif new_position[1] >= self.grid_size[1] - self.radius:
            new_position[1] = self.grid_size[1] - self.radius
            #self.velocity[1] = 0  # Stop vertical movement
            self.velocity[1] -= boundary_repulsion * (new_position[1] - (self.grid_size[1] - self.radius)) / self.radius
        
        # Set the updated position
        self.position = new_position
        
        # Ensure the cell doesn't overlap with other cells
        for other in other_cells:
            if other is not self:  # Avoid self-comparison
                displacement = self.position - other.position
                distance = np.linalg.norm(displacement)
                
                if distance < 2 * self.radius and distance > 0:
                    # Apply a small repulsion vector
                    repulsion_strength = 1  # Increase repulsion strength
                    repulsion = displacement / distance * (2 * self.radius - distance) * repulsion_strength
                    self.position += repulsion
