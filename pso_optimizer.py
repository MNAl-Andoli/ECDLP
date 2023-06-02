from collections import Iterable
import numpy as np
import ray
import redis
from tqdm import tqdm
import time
import asyncio

@ray.remote
class Particle:

    def __init__(self, random, position=[0.], velocity=[0.], dims=None):
        #self._validate(random, position, velocity, position_range, velocity_range, dims, alpha)

        self.random = random
        self.position = position
        self.velocity = velocity
        self.position_range =(-1, 1)
        self.velocity_range =(-1, 1)
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

        

    async def update(self, count, c1, c2, alpha, regular_start, gbest, fitness_fn, compare_fn):
        #print("welcome...") #, c1, c2, alpha, regular_start, gbest, fitness_fn, compare_fn

        self._update_velocity(c1, c2, alpha, gbest)
        self._update_position(count, fitness_fn, compare_fn, regular_start)
        await asyncio.sleep(0.1)   # Concurrent workload here

    def _update_velocity(self, c1, c2, alpha, gbest):
        #self.alpha = self.alpha/2 #  -  (self.alpha * 0.01)  # reduce alpha gradually
       
        wrt_pbest = c1 * np.random.rand() * (self.pbest - self.position)
        wrt_gbest = c2 * np.random.rand() * (gbest - self.position)
        self.velocity = alpha * self.velocity + wrt_pbest + wrt_gbest

    def _update_position(self, count, fitness_fn, compare_fn, regular_start):
        self.position = self.position + self.velocity + 0.01 * self.position
        
        current_ftness_value, assignment1=fitness_fn(self.position, regular_start)
        if compare_fn(current_ftness_value, self.best_fitness_value):         #fitness_fn(self.pbest)
            #print("particl number {}...the old fitness....{}, the new ....{}".format(count,  self.best_fitness_value, current_ftness_value))
            #print("particl number {}...the old position....{}, the new ....{}".format(count,  self.pbest, self.position))
            self.pbest = self.position
            self.best_fitness_value=current_ftness_value
            self.assignment=assignment1
            
    #return position of the 
    def get_position(self):
        return self.pbest
    
        #return dims of the 
    def get_dims(self):
        return self.dims

        #return fitness values 
    def get_best_fitness(self):
        return self.best_fitness_value

        #return assignment of the 
    def get_assignment(self):
        return self.assignment


######## PSO 
##init particles
def particle_init(random, num_ps,  dims):
    particles_lst= [Particle.remote(random, dims=dims) for _ in range(num_ps)]           ## need remote @@@@@@2
    gbest=ray.get(particles_lst[0].get_position.remote())            ## need remote @@@@@@2
    gbest_value=100000
    return particles_lst, gbest, gbest_value

#update _global best position

def update_gbest(particles_lst, best_fitness_lst, gbest_value, gbest, assignment):
    #get the maximum index best local fitness value
    
    min_index=np.argmin(best_fitness_lst)
    min_value=np.amin(best_fitness_lst)
    # check if the max local best fitness vallue is better than gbest
    if(min_value<gbest_value):
        #print("the old gbest....{}, the new ....{}".format(gbest_value, min_value))
        #print("the old gbest....{}, the new ....{}".format(gbest, particles_lst[min_index].get_position()))
        gbest_value=min_value
        gbest=ray.get(particles_lst[min_index].get_position.remote())          ## need remote @@@@@@2
        assignment=ray.get(particles_lst[min_index].get_assignment.remote())          ## need remote @@@@@@2
    
    
    return gbest, gbest_value, assignment

# start optimization
def optimize(num_ps,  dims, epochs, c1, c2, alpha, fitness_fn, compare_fn): 
    #initialization
    random=True  # to initialize randomly
    particles_lst, gbest, gbest_value=particle_init(random, num_ps,  dims)
    best_fitness_lst=np.zeros(num_ps)
    best_fitness_lst=best_fitness_lst + 10000 # to be maximunm value if the goal to find minumum
    
    #for i in tqdm(range(self.n_iter)):
    st=time.time()
    assignment=np.zeros(num_ps)

    #for i in range(self.n_iter):
    with tqdm(total=epochs) as pbar:
        for epoch in range(epochs):
            count=0
            for particle in particles_lst:
                particle.update.remote(count, c1, c2, alpha, epoch, gbest, fitness_fn, compare_fn) ## need remote @@@@@2
                best_fitness_lst[count]= ray.get(particle.get_best_fitness.remote())
                count+=1
                #time.sleep(0.0001) 
            gbest, gbest_value, assignment=update_gbest(particles_lst, best_fitness_lst, gbest_value, gbest, assignment)
            pbar.set_description_str(desc=str("Loss: {:.3f} ".format(gbest_value)), refresh=True)
            pbar.update(1)
    print("time consumed is: ", (time.time() -st))

    return assignment, gbest
