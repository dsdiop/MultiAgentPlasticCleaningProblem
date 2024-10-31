import numpy as np
from scipy.ndimage import gaussian_filter, convolve
import matplotlib.colors
import matplotlib.pyplot as plt
import sys
import os
from scipy.ndimage import distance_transform_edt
data_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(data_path)
algae_colormap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["dodgerblue","darkcyan", "darkgreen", "forestgreen"])
background_colormap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["sienna","sienna"])
fuelspill_colormap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["dodgerblue", "olive", "saddlebrown", "indigo"])
macroplastic_colormap = matplotlib.colors.LinearSegmentedColormap.from_list("", [(0,"dodgerblue"),(0.1, 'green'), (0.5, 'yellow'), (1, 'red')])


class macro_plastic:

    def __init__(self, grid: np.ndarray, dt = 0.1, max_number_of_pollution_spots = 3, total_trash_elements = 100, seed = 0) -> None:
        """ Generador de ground truths de plásticos con dinámica """
        self.seed = seed
        self.rng = np.random.default_rng(seed=self.seed) # random number generator, it's better than set a np.random.seed() (https://builtin.com/data-science/numpy-random-seed)
        self.rng_seed_for_steps = np.random.default_rng(seed=self.seed+1)
        self.rng_steps = np.random.default_rng(seed=self.rng_seed_for_steps.integers(0, 1000000))  
        # Random generators declaration #
        self.rng_wind_direction = np.random.default_rng(seed=self.seed)
        self.rng_number_of_trash_elements = np.random.default_rng(seed=self.seed)
        self.rng_trash_positions_MVN = np.random.default_rng(seed=self.seed)
        self.rng_pollution_spots_number = np.random.default_rng(seed=self.seed)
        self.rng_pollution_spots_locations_indexes = np.random.default_rng(seed=self.seed)      
        # Creamos un mapa vacio #
        self.map = np.zeros_like(grid)
        self.grid = grid
        self.particles = None
        self.starting_point = None
        self.visitable_positions = np.column_stack(np.where(grid == 1))
        self.fig = None
        self.dt = dt
        self.max_number_of_pollution_spots = max_number_of_pollution_spots
        self.total_trash_elements = total_trash_elements
        
        distances, self.closest_indices = distance_transform_edt(grid == 0, return_indices=True)
        
        self.discretized_particles = np.array([])
    def reset(self):
        #self.in_bound_particles = np.array([])
        self.pollution_spots_number = self.rng_pollution_spots_number.integers(1, self.max_number_of_pollution_spots+1)
        #starting_points = [np.array((self.rng.integers(self.map.shape[0]/6, 5*self.map.shape[0]/6), self.rng.integers(self.map.shape[1]/6, 5* self.map.shape[1]/6)))
        #                   for _ in range(self.pollution_spots_number)]
        
        starting_points = self.rng_pollution_spots_locations_indexes.choice(np.arange(0, len(self.visitable_positions)), self.pollution_spots_number, replace=False)
        # number_of_trash_elements_in_each_spot = self.rng_number_of_trash_elements.normal(loc=0, 
        #                                                                               scale=self.max_number_of_trash_elements_per_spot, 
        #                                                                               size=self.pollution_spots_number).round().astype(int)
        # self.number_of_trash_elements_in_each_spot = np.clip(np.abs(number_of_trash_elements_in_each_spot),int(100/number_of_trash_elements_in_each_spot.shape[0]), self.max_number_of_trash_elements_per_spot)
        #total_trash_elements = 100
        number_of_trash_elements_in_each_spot = self.rng_number_of_trash_elements.normal(loc=0, 
                                                  scale=self.total_trash_elements, 
                                                  size=self.pollution_spots_number).round().astype(int)
        number_of_trash_elements_in_each_spot = np.clip(np.abs(number_of_trash_elements_in_each_spot), 1, self.total_trash_elements)
        number_of_trash_elements_in_each_spot = (number_of_trash_elements_in_each_spot / number_of_trash_elements_in_each_spot.sum()) * self.total_trash_elements
        self.number_of_trash_elements_in_each_spot = number_of_trash_elements_in_each_spot.round().astype(int)
        self.number_of_trash_elements_in_each_spot[0] += self.total_trash_elements - self.number_of_trash_elements_in_each_spot.sum() # take the first element and add the difference if there is any
        cov = 7.0
        self.particles = self.rng_trash_positions_MVN.multivariate_normal(self.visitable_positions[starting_points[0]], np.array([[cov, 0.0],[0.0, cov]]),size=(self.number_of_trash_elements_in_each_spot[0],)) 
        for i in range(1, self.pollution_spots_number):
            self.particles = np.vstack(( self.particles, self.rng.multivariate_normal(self.visitable_positions[starting_points[i]], np.array([[cov, 0.0],[0.0, cov]]),size=(self.number_of_trash_elements_in_each_spot[i],))))
        self.particles = np.clip(self.particles, 0, np.array(self.map.shape)-1)
        self.particles = np.array([self.keep_inside_navigable_zone(particle) for particle in self.particles if self.is_inside_map(particle)])
        #self.inbound_particles = np.array([self.keep_inside_navigable_zone(particle) for particle in self.particles])
        self.discretize_map()


        # New seed for steps #
        self.wind_direction = self.rng_wind_direction.uniform(low=-1.0, high=1.0, size=2)
        self.rng_steps = np.random.default_rng(seed=self.rng_seed_for_steps.integers(0, 1000000))
        
        return self.map

        
    def apply_current_field(self, particle):

        current_movement = self.wind_direction + 0.1*self.rng_steps.uniform(low=-1.0, high=1.0, size=2)
        new_particle = np.clip(particle + self.dt*current_movement, 0, np.array(self.map.shape)-1)
        
        return new_particle if self.is_inside_map(new_particle) else None

    def keep_inside_navigable_zone(self, particle):
        
        if self.grid[np.round(particle[0]).astype(int),np.round(particle[1]).astype(int)] == 1:
            return particle
        else:
            particle = np.round(particle).astype(int)
            nearest_x, nearest_y = self.closest_indices[:, particle[0], particle[1]]
            return np.array([nearest_x, nearest_y])
        
    def is_inside_map(self, particle):
            #particle = particle.astype(int)
            if particle[0] >= 0 and particle[0] < self.map.shape[0] and  particle[1] >= 0 and particle[1] < self.map.shape[1]:
    
                return True
            else:
                return False   
             
    def step(self):

        particles = np.array([self.apply_current_field(particle) for particle in self.particles])
        self.particles = np.array([self.keep_inside_navigable_zone(particle) for particle in particles if particle is not None])
        self.discretize_map()
        return self.map

    def render(self):
        
        f_map = self.map
        f_map[self.grid == 0] = np.nan

        if self.fig is None:
            self.fig, self.ax = plt.subplots(1,1)
            self.d = self.ax.imshow(f_map, cmap = macroplastic_colormap)
            
            background = self.grid.copy()
            background[background == 1] = np.nan
            self.ax.imshow(background, cmap=background_colormap)
            
        else:
            self.d.set_data(f_map)

        self.fig.canvas.draw()
        plt.pause(0.01)
    
    def read(self):
        self.discretize_map()
        return self.map
    
    def clean_particles(self, position, n_particles):
        "Given a position and the number of particles to clean, it removes the particles from the list of particles"
        # We see what particles are inside the position
        particles_to_remove = []
        for i, discretized_particle in enumerate(self.discretized_particles):
            if np.all(discretized_particle == position):
                particles_to_remove.append(i)
            if len(particles_to_remove) == n_particles:
                break
        # We remove the particles
        self.particles = np.delete(self.particles, particles_to_remove, axis=0)
        self.discretize_map()
        return len(particles_to_remove)
    
    def discretize_map(self):
        self.map[:,:] = 0.0
        self.discretized_particles = np.zeros_like(self.particles).astype(int)
        for i, particle in enumerate(self.particles):
            self.discretized_particles[i] = np.round(particle).astype(int)
            self.map[self.discretized_particles[i][0], self.discretized_particles[i][1]] += 1.0
            
    #def apply_gaussian_filter_and_normalize(self):    
        self.filtered_map = gaussian_filter(self.map, 5, mode = 'constant', cval=0, radius = None) * self.grid
        if np.max(self.filtered_map) == 0:
            self.normalized_filtered_map = np.zeros_like(self.filtered_map)
        else:
            self.normalized_filtered_map = (self.filtered_map-np.min(self.filtered_map))/(np.max(self.filtered_map)-np.min(self.filtered_map))

if __name__ == '__main__':

    import matplotlib.pyplot as plt

    gt = macro_plastic(np.genfromtxt(f'{data_path}/Maps/malaga_port.csv', delimiter=','), dt=0.2, seed=10)

    m = gt.reset()
    gt.render()

    for _ in range(50000):

        #m = gt.reset()
        gt.step()
        print(str(_))

        gt.render()
        plt.pause(0.5)

    


        
        
        