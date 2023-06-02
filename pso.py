from collections import Iterable
import numpy as np
from .particle import Particle
from tqdm import tqdm
import ray
import time

class ParticleSwarmOptimizer:

    def __init__(self, particle_cls, c1, c2, alpha, n_particles,
                 fitness_fn, compare_fn, n_iter=1, dims=None,
                 random=True, particles_list=None, position_range=None,
                 velocity_range=None):
        self.particle_cls = particle_cls
        self.c1 = c1
        self.c2 = c2
        self.n_particles = n_particles
        self.n_iter = n_iter
        self.fitness_fn = fitness_fn
        self.compare_fn = compare_fn
        self.position_range = position_range
        self.velocity_range = velocity_range
        self.dims = dims
        self.random = random
        self.particles_list = particles_list
        self._init_particles_list()
        self.alpha=alpha
        print("c1 {}, c2 {}, alpha 'w' : {}".format(self.c1, self.c2, self.alpha))
        

    def _get_fitness(self, position):
        return self.fitness_fn(position)

    def _update_gbest(self):
        for particle_i in self.particles_list:
            #l1, l2 = self._get_fitness(particle_i.pbest), self._get_fitness(self.gbest)
            #l1, l2 = particle_i.best_fitness_value, self.gbest_fitness_value
            l1, l2 = ray.get(particle_i.get_best_fitness.remote()), self.gbest_fitness_value #@@@@@@@@@@@@@@@@@@@remote
            if self.compare_fn(l1, l2):
                #self.gbest = particle_i.position
                self.gbest = ray.get(particle_i.get_position.remote())   #@@@@@@@@@@@@@@@@@@@remote
                self.gbest_fitness_value=l1
                

    def _init_particles_list(self):
        if self.random:
            self.particles_list = []

            for i in range(self.n_particles):
                particle = self.particle_cls.remote(self.random, position_range=self.position_range,         #@@@@@@@@@@@@remote
                                             velocity_range=self.velocity_range, dims=self.dims)
                self.particles_list.append(particle)

        #self.gbest = self.particles_list[0].get_position()
        obj = self.particles_list[0].get_position.remote()#@@@@@@@@@@@@@@@remote
        self.gbest=ray.get(obj)
        #print("gbest1c 0 {}, gbest1 10 {}, gbest 0 {}, gbest 10 {}".format(gbest1[0], gbest1[10], self.gbest[0], self.gbest[10]))


        self.gbest_fitness_value=  100000  # to initilaize large number of best paricle (in minimize, min valu in maximize) of best values to avoid call fitness function more times
        
        self._update_gbest()

        #self.dims = self.particles_list[0].dims
        self.dims = ray.get(self.particles_list[0].get_dims.remote())#@@@@@@@@@@@@@@@@@@@remote

    def optimize(self, epochs):
        #for i in tqdm(range(self.n_iter)):
        st=time.time()
        assignment=[]
        #for i in range(self.n_iter):
        with tqdm(total=epochs) as pbar:
            for epoch in range(epochs):
                for particle in self.particles_list:
                    particle.update.remote(self.c1, self.c2, self.alpha, epoch, self.gbest, self.fitness_fn, self.compare_fn)
                self._update_gbest()
                pbar.set_description_str(desc=str("Loss: {:.3f} ".format(self.gbest_fitness_value)), refresh=True)
                pbar.update(1)
        assignment=ray.get(particle.get_assignment.remote())
        #self.gbest
        #particle.velocity
        print("time consumed is: ", (time.time() -st))
        #print( "Gbest", self.gbest_fitness_value)
        return assignment
