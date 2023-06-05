import ray
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import time
ray.init()
 
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')

gpu_fraction=0.25
cpu_fraction=3
num_models=4
partition=True
local_epochs=200
PSO_epochs=200
current_global_loss=0
#ray.nodes()

def load_data():
    x_train=np.load("X_train.npy")
    #=====normalize x_train. z-score
    scaler=StandardScaler()
    scaler.fit(x_train)
    x_train=scaler.transform(x_train)
    #=====
    y_train=np.load("y_train.npy")
    
    return x_train , y_train

def create_keras_model(input_dim):
    from tensorflow import keras
    from tensorflow.keras import layers

    # to allow GPU fraction
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.25,
                                             allow_growth=True)        
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
    tf.compat.v1.keras.backend.set_session(sess)
    
    
    model = keras.Sequential()
    # Adds a densely-connected layer with 64 units to the model:
    model.add(layers.Dense(128, activation="relu", input_shape=(input_dim, )))
    # Add another:
    model.add(layers.Dense(64, activation="relu"))
    # Add a softmax layer with 10 output units:
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

# Use CPU and GPU for parallization
#@ray.remote(num_gpus=1)
#@ray.remote
@ray.remote (num_cpus=cpu_fraction, num_gpus=gpu_fraction)
class train_model(object):
    def __init__(self, dataset_id, labels_id, index_model, num_models):
        
        #use the whole dataset for the evaluation
        self.whole_dataset=dataset_id
        self.wohle_labels=labels_id
        #with partitions

        if (partition):
            size_partition=int(dataset_id.shape[0]/num_models)
            start=(index_model * size_partition) 
            end= start + size_partition
            self.dataset= dataset_id [start:end,:]
            self.labels=  labels_id[start:end]
        #without partitions
        else:
            self.dataset=dataset_id
            self.labels=labels_id
            
        input_dim= self.dataset.shape[1]
        
        self.model = create_keras_model(input_dim)
        


    def train(self):
        history = self.model.fit(self.dataset, self.labels, epochs=local_epochs, batch_size=256, verbose=False)
        return history.history['accuracy']

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        # Note that for simplicity this does not handle the optimizer state.
        self.model.set_weights(weights)
    
    def evaluate_for_PSO(self, weights):
        self.model.set_weights(weights)
        '''_, loss_func=self.model.evaluate(self.whole_dataset, self.wohle_labels)
        return loss_func'''
        
        history = self.model.fit(self.whole_dataset, self.wohle_labels, verbose=False)
        return history.history['accuracy']
        
    def model_save(self):
        self.model.save('parallel_model.h5')
        print("done...")
# train models in parallel 
def train_in_parallel(actors, num_models):
    obj_train_ids=[actors[m].train.remote() for m in range(num_models)]

#=================================

class PSO(object): 
    def __init__(self, init_params):
        self.local_position=init_params
        self.velocity=init_params
        self.global_position=init_params[0]
        self.best_model=0
        self.c1=0.3
        self.c2=0.1
        self.alpha=0.1
    
    # should do this in parallel and perform inside te main class
    def update_local_best(self, new_local_weights, index_model):
                             
        """update velocity"""
        wrt_pbest =[((self.c1 * np.random.uniform()) *
                    (self.local_position[index_model][layer] - new_local_weights[index_model][layer]))
                    for layer in range(len(new_local_weights[index_model]))]
        
        #wrt_gbest=wrt_pbest
        wrt_gbest =[((self.c2 * np.random.uniform()) *
                    (self.global_position[layer] - new_local_weights[index_model][layer]))
                    for layer in range(len(new_local_weights[index_model]))]
        

        wrt_pbest_gbest=[(wrt_pbest[layer]+ wrt_gbest[layer])
                    for layer in range(len(wrt_pbest))]
        #+ (wrt_pbest_gbest)
        self.velocity[index_model]= [((self.alpha * self.velocity[index_model] [layer]) + (wrt_pbest_gbest[layer]))
                            for layer in range(len(self.velocity[index_model]))]     
        
        #update new positions 'weights'
        new_local_weights[index_model]= [(new_local_weights [index_model][layer] + self.velocity [index_model][layer])
                                for layer in range(len(new_local_weights[index_model]))]
        
        #now calculate fitness function again, and select the best one, also merge wie average
        

    # calculate average weights
    def calc_avg_weights(self, weights):
        #initilize it
        average_weights=weights[0]

        for m in range(1,num_models):
            for l in range(len(weights[m])):
                layer=weights[m][l]
                average_weights[l]= average_weights[l] + layer        

        #print(average_weights[5])    
        for l in range(len(average_weights)):
            average_weights[l]= (average_weights[l] /num_models)
        
        #evaluate the average weights
        loss_avg=ray.get(actors[0].evaluate_for_PSO.remote(average_weights))  
        
        return average_weights, loss_avg


    
    def update_global_best(self, new_local_weights, loss_func_all_epochs):
        #get the global best loss function
        current_global_loss=0
        loss_func=np.zeros(num_models)
        
        
        #get loss function of current local weights
        for m in range(num_models):
            loss_func[m] = np.max(loss_func_all_epochs[m])
        if((np.max(loss_func)) > current_global_loss):
            self.best_model=np.argmax(loss_func)
            current_global_loss=np.max(loss_func)
            #set global position
            self.global_position=new_local_weights[self.best_model]
            
        #update the global best by the average weights if it is better than the local best
        #calc avg of weights and loss function
        average_weights, loss_avg=self.calc_avg_weights(weights)
        
        if(loss_avg>current_global_loss):
            current_global_loss=loss_avg
            #set global position
            self.global_position=avg_weight
            

        
    def calc_fitness_func(self, loss_func_all_epochs, new_local_weights, actors, i):
                    
        #get local and  global weights
        self.update_global_best(new_local_weights,loss_func_all_epochs)
        for m in range(num_models):
            self.update_local_best(new_local_weights,m)
        #evaluate the models on all dataset after updating weights 
        loss_values=ray.get([actors[m].evaluate_for_PSO.remote(new_local_weights[m]) for m in range(num_models)])
        print(loss_values)
        actors[i].save('my_actor' + str(i+1)+ '.h5')
        #set global position to the best one

t1=time.time()

#load data to global memory
dataset, labels=load_data()

dataset_id =ray.put(dataset)
labels_id =ray.put(labels)

del dataset
del labels



actors=[train_model.remote(dataset_id, labels_id, i, num_models) for i in range(num_models)] 

for i in range(PSO_epochs):
    train_in_parallel(actors, num_models)
    results=ray.get([actors[m].train.remote() for m in range(num_models)])
 
    weights_ids=[actors[m].get_weights.remote() for m in range(num_models)]
    weights=ray.get(weights_ids)

    pso1=PSO(weights)
    pso1.calc_fitness_func(results, weights,actors, i)
    
models = actors

#load model
loaded_models=[]
for i in range(len(models)):
    loaded_model=keras.models.load_model("my_model" + str(i+1) + ".h5")
    loaded_models.append(loaded_model)
    loaded_models[i].compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    
    # make an ensemble prediction for multi-class classification
def ensemble_predictions(models, testX):
    labels = []    
    for m in models:
        predicts = (m.predict(x_test) > 0.5).astype("int32")
        #redicts=[list(i) for i in predicts]
        labels.append(predicts)

    # Ensemble with voting
    labels = np.array(labels)
    labels = np.transpose(labels)

    labels = scipy.stats.mode(labels, 2)[0]

    labels = np.squeeze(labels)
    
    return labels

# You can add more models like inception, vgg-16/vgg-19 etc. to improve overall accuracy during ensemble
t2=time.time()
print("time consumed is: ", t2-t1)

#test
#load model
from tensorflow import keras
import numpy as np
#ray.shutdown()
loaded_model=keras.models.load_model("parallel_model.h5")
loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

x_test=np.load("X_test.npy")

#====z_score normalization
scaler2=StandardScaler()
scaler2.fit(x_test)
x_test=scaler2.transform(x_test)
#====
y_test=np.load("y_test.npy")


predict_labels=ensemble_predictions(loaded_models, x_test)
Enemble_accuracy = accuracy_score(y_test,predict_labels)

for i in range(len(loaded_models)):
    _, accuracy2 = loaded_models[i].evaluate(x_test, y_test)
    print('Accuracy: %.2f' % (accuracy2*100))
print ("Ensemble accuracy: ", Enemble_accuracy)
