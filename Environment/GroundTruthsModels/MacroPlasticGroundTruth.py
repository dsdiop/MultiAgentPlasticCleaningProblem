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

    def __init__(self, grid: np.ndarray, dt = 0.2, max_number_of_pollution_spots = 3, max_number_of_trash_elements_per_spot = 100, seed = 0) -> None:
        """ Generador de ground truths de algas con dinámica """
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
        self.max_number_of_trash_elements_per_spot = max_number_of_trash_elements_per_spot
        
        distances, self.closest_indices = distance_transform_edt(grid == 0, return_indices=True)
        
        #self.contour_currents_x = convolve(self.grid, np.array([[0,0,0,0,0],[0,0,0,0,0],[0,0,1,-1,-2],[0,0,0,0,0],[0,0,0,0,0]]), mode='constant')
        #self.contour_currents_y = convolve(self.grid, np.array([[0,0,0,0,0],[0,0,0,0,0],[0,0,1,0,0],[0,0,-1,0,0],[0,0,-2,0,0]]), mode='constant')
        self.contour_currents_x = convolve(self.grid, np.array([[0,0,0],[0,1,-1],[0,0,0]]), mode='constant')*2
        self.contour_currents_y = convolve(self.grid, np.array([[0,0,0],[0,1,0],[0,-1,0]]), mode='constant')*2
    def reset(self):
        #self.in_bound_particles = np.array([])
        self.pollution_spots_number = self.rng_pollution_spots_number.integers(1, self.max_number_of_pollution_spots+1)
        #starting_points = [np.array((self.rng.integers(self.map.shape[0]/6, 5*self.map.shape[0]/6), self.rng.integers(self.map.shape[1]/6, 5* self.map.shape[1]/6)))
        #                   for _ in range(self.pollution_spots_number)]
        
        starting_points = self.rng_pollution_spots_locations_indexes.choice(np.arange(0, len(self.visitable_positions)), self.pollution_spots_number, replace=False)
        number_of_trash_elements_in_each_spot = self.rng_number_of_trash_elements.normal(loc=0, 
                                                                                      scale=self.max_number_of_trash_elements_per_spot, 
                                                                                      size=self.pollution_spots_number).round().astype(int)
        self.number_of_trash_elements_in_each_spot = np.clip(np.abs(number_of_trash_elements_in_each_spot),10, self.max_number_of_trash_elements_per_spot)
        #number_of_trash_elements_in_each_spot[number_of_trash_elements_in_each_spot <= 0] = 10 # minimum number of trash elements in a spot
        cov = 7.0
        self.particles = self.rng_trash_positions_MVN.multivariate_normal(self.visitable_positions[starting_points[0]], np.array([[cov, 0.0],[0.0, cov]]),size=(self.number_of_trash_elements_in_each_spot[0],)) 
        for i in range(1, self.pollution_spots_number):
            self.particles = np.vstack(( self.particles, self.rng.multivariate_normal(self.visitable_positions[starting_points[i]], np.array([[cov, 0.0],[0.0, cov]]),size=(self.number_of_trash_elements_in_each_spot[i],))))
        self.particles = np.clip(self.particles, 0, np.array(self.map.shape)-1)
        self.particles = np.array([self.keep_inside_navigable_zone(particle) for particle in self.particles if self.is_inside_map(particle)])
        #self.inbound_particles = np.array([self.keep_inside_navigable_zone(particle) for particle in self.particles])
        
        for particle in self.particles:
            self.map[np.round(particle[0]).astype(int), np.round(particle[1]).astype(int)] += 1.0

        #self.algae_map = gaussian_filter(self.map, 0.8)
        # New seed for steps #
        self.wind_direction = self.rng_wind_direction.uniform(low=-1.0, high=1.0, size=2)
        self.rng_steps = np.random.default_rng(seed=self.rng_seed_for_steps.integers(0, 1000000))
        
        return self.map

        
    def apply_current_field(self, particle):

        current_movement = self.wind_direction + self.rng_steps.random()
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
        self.map[:,:] = 0.0
        for particle in self.particles:
            self.map[np.round(particle[0]).astype(int), np.round(particle[1]).astype(int)] += 1.0
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

        return self.map

if __name__ == '__main__':

    import matplotlib.pyplot as plt

    gt = macro_plastic(np.genfromtxt(f'{data_path}/Maps/malaga_port.csv', delimiter=','), dt=0.2, seed=3)

    m = gt.reset()
    gt.render()

    for _ in range(50000):

        #m = gt.reset()
        gt.step()
        print(str(_))
        gt.render()
        plt.pause(0.5)

    


        
        
        