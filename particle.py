from collections import Iterable
import numpy as np
import ray
import redis

@ray.remote
class Particle:

    def __init__(self, random, position=[0.],
                 velocity=[0.], position_range=None,
                 velocity_range=None, dims=None):
        #self._validate(random, position, velocity, position_range, velocity_range, dims, alpha)

        self.random = random
        self.position = position
        self.velocity = velocity
        self.position_range = position_range
        self.velocity_range = velocity_range
        self.dims = dims
        self.assignment=[]
        self._init_particle()

        self.pbest = self.position
        
    def _init_particle(self):
        if self.random:
            self.position = np.random.uniform(low=self.position_range[0],
                                              high=self.position_range[1],
                                              size=(self.dims,))
            self.velocity = np.random.uniform(low=-abs(self.velocity_range[1]-self.velocity_range[0]),
                                              high=abs(self.velocity_range[1]-self.velocity_range[0]),
                                              size=(self.dims,))
        else:
            self.position = np.asarray(position)
            self.velocity = np.asarray(velocity)
            self.dims = self.position.shape[0]
            
        self.best_fitness_value=100000  # to save values of pbest and gbest instead of calculate again

        

    def update(self, c1, c2, alpha, regular_start, gbest, fitness_fn, compare_fn):
        #print("welcome...") #, c1, c2, alpha, regular_start, gbest, fitness_fn, compare_fn

        self._update_velocity(c1, c2, alpha, gbest)
        self._update_position(fitness_fn, compare_fn, regular_start)

    def _update_velocity(self, c1, c2, alpha, gbest):
        #self.alpha = self.alpha/2 #  -  (self.alpha * 0.01)  # reduce alpha gradually
       
        wrt_pbest = c1 * np.random.rand() * (self.pbest - self.position)
        wrt_gbest = c2 * np.random.rand() * (gbest - self.position)
        self.velocity = alpha * self.velocity + wrt_pbest + wrt_gbest

    def _update_position(self, fitness_fn, compare_fn, regular_start):
        self.position = self.position + self.velocity + 0.01 * self.position
        
        current_ftness_value, assignment1=fitness_fn(self.position, regular_start)
        
        #print("fitness_fn(self.pbest)", fitness_fn(self.pbest), " self.best_fitness_value", self.best_fitness_value)
        #if compare_fn(fitness_fn(self.position), fitness_fn(self.pbest)):
        if compare_fn(current_ftness_value, self.best_fitness_value):         #fitness_fn(self.pbest)
            self.pbest = self.position
            self.best_fitness_value=current_ftness_value
            self.assignment=assignment1
            
    #return position of the 
    def get_position(self):
        return self.position
    
        #return dims of the 
    def get_dims(self):
        return self.dims

        #return fitness values 
    def get_best_fitness(self):
        return self.best_fitness_value

        #return assignment of the 
    def get_assignment(self):
        return self.assignment

    
    def __repr__(self):
        return '<Particle: dims={} random={}>'.format(self.dims, self.random)
