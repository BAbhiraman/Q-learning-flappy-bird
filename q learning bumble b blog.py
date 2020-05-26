# -*- coding: utf-8 -*-
"""
Created on Tue May 26 19:28:36 2020

@author: Bhaskar Abhiraman
Q-learning wrapper based on: https://towardsdatascience.com/reinforcement-learning-w-keras-openai-dqns-1eed3a5338c
Requires graphics.py by John Zelle: https://mcsp.wartburg.edu/zelle/python/

"""

import numpy as np
import random
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pylab as pl
from collections import deque

import graphics as g
import time


class DQN:
    def __init__(self, old_model = None):
        self.memory  = deque(maxlen=2000)
        
        self.gamma = 0.99
        self.epsilon = 0.1 
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.995
        self.learning_rate = 0.005
        self.tau = .125
        self.old_model = old_model
        self.model        = self.create_model()
        self.target_model = self.create_model()

    def create_model(self):
        if self.old_model is not None:
            model = self.old_model
        else:
            model   = Sequential()
            model.add(Dense(6, input_dim=2, activation="relu")) #input_dim = shape of STATE
            model.add(Dense(2)) #NUMBER OF ACTIONS
            model.compile(loss="mean_squared_error",
                optimizer=Adam(lr=self.learning_rate))
        return model

    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            return int(np.random.random() < 1/10) #RANDOM ACTION (BIAS AGAINST JUMPS)
        return np.argmax(self.model.predict(state)[0])

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        batch_size = 32
        if len(self.memory) < batch_size: 
            return
        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample
            target = self.target_model.predict(state)
            if done:
                target[0][action] = reward
            else:
                Q_future = max(self.target_model.predict(new_state)[0])
                target[0][action] = reward + Q_future * self.gamma
            self.model.fit(state, target, epochs=1, verbose=0)

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def save_model(self, fn):
        self.model.save(fn)

def main():
    game    = Game(graph = False)
    trials  = 100000
    trial_len = 10000
    dqn_agent = DQN()
    for trial in range(trials):
        game = Game(graph = False)
        cur_state = game.game_vars()
        cur_state = np.asarray(cur_state).reshape(1,2)
        for step in range(trial_len):
            action = dqn_agent.act(cur_state)
            new_state, reward, done, fr_sc_jmp = game.action_step(action)
            frame, score, jumps = fr_sc_jmp
            new_state = np.asarray(new_state)
            new_state = new_state.reshape(1,2)
            dqn_agent.remember(cur_state, action, reward, new_state, done)
            
            dqn_agent.replay()       # internally iterates default (prediction) model
            dqn_agent.target_train() # iterates target model

            cur_state = new_state
            if done:
                break
        if score < Game.thresh: #thresh
            print("Failed to complete in trial {}".format(trial))
            print("frame: {}, score: {}, jumps: {}".format(frame, score, jumps))
        else:
            print("Completed in {} trials".format(trial))
            dqn_agent.save_model("successes/success{}.model".format(int(np.random.random()*1000)))
            break
        
class Game:
    dt = 1/20    
    a = 11
   
    sq_len = 20
    pipe_dx = 50
    pipe_dy = 250 #was 250 for success 1-3 #success final can run 165
    pipe_v = 70
    pipe_space = 250
    pipe_buff = 50
    pipe_range = (pipe_buff + pipe_dy/2, 500-(pipe_buff + pipe_dy/2))
    
    #Variables for the network drawing
    num_nodes = [2,6,2]
    num_lay = len(num_nodes)
    x0 = 0
    space = 600/(num_lay+1)
    y0 = 100 #500 
    ymax = 400 #800
    
    thresh = 10 #number of points to win
    
    def __init__(self, graph = True, success_model = None, pipe_dy = 250, frame_rate = 35):
        self.v = 0
        self.graph = graph
        self.success_model = success_model
        if self.graph:
            self.win = g.GraphWin("Bumble B", 600, 500, autoflush = False)
            self.win.setBackground(g.color_rgb(69, 140, 255))
        else:
            self.win = None
        self.rec = g.Rectangle(g.Point(5,20), g.Point(5+Game.sq_len, 20+Game.sq_len))
        self.rec.setFill("Yellow")
        self.rec.setOutline('Yellow')
        self.score = 0
        self.tscore = 0
        self.t = 0
        self.frame = 0
        self.txt = None
        self.over = False
        self.jumps = 0
        self.neur_vals_disc = [2*[-100],6*[-100],2*[-100]]
        self.neur_circs = [2*[None],6*[None],2*[None]]
        self.circ_temp = None
        self.pipe_dy = pipe_dy
        self.frame_rate = frame_rate
        pipe1 = Game.Pipe(Game.pipe_dx, 240, self.pipe_ran(), Game.pipe_v, self.pipe_dy, self.win)
        pipe2 = Game.Pipe(Game.pipe_dx, 240 + 1*Game.pipe_space, self.pipe_ran(), Game.pipe_v, self.pipe_dy, self.win)
        pipe3 = Game.Pipe(Game.pipe_dx, 240 + 2*Game.pipe_space, self.pipe_ran(), Game.pipe_v, self.pipe_dy, self.win)
        self.pipes = [pipe1, pipe2, pipe3]
        
        if self.graph:
            self.rec.draw(self.win)
        
        for pipe in self.pipes:
            if self.graph:
                pipe.draw(self)
        
        if (self.success_model is not None) and self.graph:
            self.neuron_col = pl.cm.inferno
            self.neuron_col_r = pl.cm.inferno_r
            self.hidden_layers = [None,None]
            for i in range(2):
                self.hidden_layers[i] = keras.backend.function(
                [success_model.layers[0].input],  # we will feed the function with the input of the first layer
                [success_model.layers[i].output]) # we want to get the output of the ith layer   
               
                weights = self.success_model.get_weights()
                
                def draw_network_weights(weights):
                    white_rec = g.Rectangle(g.Point(0,500), g.Point(600,800))
                    white_rec.setFill('white')
                    white_rec.draw(self.win)
                    weight_col = pl.cm.PuOr
                    weights0 = weights[0]
                    weights1 = weights[2]
                    def draw_weights(sweights, lay1, lay2, colscale = 1):
                        wmax = np.max(np.abs(sweights))
                        x= [Game.x0 + (lay1+1)*Game.space,Game.x0 + (lay2+1)*Game.space]
                        m,n = np.shape(sweights)
                        for i in range(m):
                            for j in range(n):
                                if sweights[i,j] != 0:
                                    y1 = np.linspace(Game.y0,Game.ymax,Game.num_nodes[lay1]+2)[1:-1][i]
                                    y2 = np.linspace(Game.y0,Game.ymax,Game.num_nodes[lay2]+2)[1:-1][j]
                                    col = (255*np.asarray(weight_col(0.5 + 0.5*sweights[i,j]*colscale/wmax))[0:3]).astype(int)
                                    line = g.Line(g.Point(x[0], y1), g.Point(x[1], y2))
                                    line.setWidth(3.5)
                                    line.setFill(g.color_rgb(*col))
                                    line.draw(self.win)
    
                    draw_weights(weights0, 0, 1, colscale = 1)
                    draw_weights(weights1, 1, 2, colscale = 1)
                draw_network_weights(weights)
        
    class Pipe:
    
        def __init__(self, dx,xi, y, v, dy, win):
            self.dx = dx
            self.y = y
            self.v = v
            self.top = g.Rectangle(g.Point(xi,0), g.Point(xi + dx, y - dy/2))
            self.bot = g.Rectangle(g.Point(xi,500), g.Point(xi + dx, y + dy/2))
            self.top.setFill(g.color_rgb(129, 232, 88))
            self.top.setOutline(g.color_rgb(129, 232, 88))
            self.bot.setFill(g.color_rgb(129, 232, 88))
            self.bot.setOutline(g.color_rgb(129, 232, 88))
            self.win = win
        def move(self):
            self.top.move(-self.v*Game.dt, 0)
            self.bot.move(-self.v*Game.dt, 0)
        def draw(self, game):
            if game.graph:
                self.top.draw(self.win)
                self.bot.draw(self.win)
        def check_kill(self, game):
            if self.top.getP2().getX() < 0:
                if game.graph: 
                    self.top.undraw()
                    self.bot.undraw()
                return True
        def get_loc(self):
            return (self.top.getP1().getX(), self.top.getP2().getY(), self.bot.getP2().getY())
        def check_coll(self, game):
            cube_top_y, cube_bot_y = (game.rec.getP1().getY(), game.rec.getP2().getY())
            cube_xr = game.rec.getP2().getX()
            pipe_xl, pipe_top_y, pipe_bot_y = self.get_loc()
            if (pipe_top_y > cube_top_y and cube_top_y > 0) and (cube_xr > pipe_xl and (cube_xr - Game.sq_len) < (pipe_xl + Game.pipe_dx)):
                return True
            elif (cube_bot_y > pipe_bot_y and cube_bot_y < 500) and (cube_xr > pipe_xl and (cube_xr - Game.sq_len) < (pipe_xl + Game.pipe_dx)):
                return True
            else:
                return False
            
    def check_bounds(self):    
        cube_top_y = self.rec.getP1().getY()
        cube_bot_y = self.rec.getP2().getY()
        return (cube_top_y < 0 or cube_bot_y > 500)
            
    def run(self, robot = None):
        while not self.over:
            _, _, _, _ = self.step(robot)
        if self.graph:
            time.sleep(1)
            self.win.close()    # Close window when done
        return(self.tscore)
        
    def pipe_ran(self): #generate random pipe location
        pmin, pmax = Game.pipe_range
        return np.random.random()*(pmax-pmin) + pmin
    
    def game_vars(self):
        cube_top_y = self.rec.getP1().getY()
        cube_bot_y = self.rec.getP2().getY()
        pipe_tops = [None, None, None]
        pipe_bots = [None, None, None]
        pipe_edge = [None, None, None]
        for i, pipe in enumerate(self.pipes):
            pipe_edge[i], pipe_tops[i], pipe_bots[i] = pipe.get_loc()
        pipe_top_dist = cube_top_y - np.asarray(pipe_tops)
        pipe_edge_dist = np.asarray(pipe_edge) - self.rec.getP2().getX()
        pipe_bot_dist = np.asarray(pipe_bots) - cube_bot_y
        pipe_mid_dist = (cube_top_y + cube_bot_y)/2 - (np.asarray(pipe_tops)+np.asarray(pipe_bots))/2
        inds = np.argsort(pipe_edge_dist) 
        pipe_edge_dist = pipe_edge_dist[inds]
        pipe_top_dist = pipe_top_dist[inds]
        pipe_bot_dist = pipe_bot_dist[inds]
        pipe_mid_dist = pipe_mid_dist[inds]

        self.tscore = self.frame*self.dt + 3*self.score
        if self.graph:
            s = "{}".format(int(self.score))
            try: 
                self.txt.undraw()
            except:
                pass 
            self.txt = g.Text(g.Point(567,20), s)
            self.txt.setStyle('bold')
            self.txt.setSize(20)
            self.txt.draw(self.win)
        #return [cube_v, cube_top_y, 500-cube_bot_y] + list(pipe_edge_dist) + list(pipe_top_dist) + list(pipe_bot_dist)
        return [pipe_mid_dist[0], pipe_edge_dist[0]]

    def draw_neuron_help(self, neuron_layer, lay, markersize = 7, colscale = 1, first = False):
        x = Game.x0 + (lay+1)*Game.space
        ys = np.linspace(Game.y0,Game.ymax,Game.num_nodes[lay]+2)[1:-1]
        if not first:
            mx = max(np.max(neuron_layer),0.01)
            cmap = self.neuron_col
        else:
            mx = 250
            cmap = self.neuron_col_r
            neuron_layer[0] += 125
        for i in range(len(neuron_layer)):
            disc = 100
            val = int(disc*neuron_layer[i]*colscale/mx)  #you can choose to only update 
            if True: #val != self.neur_vals_disc[lay][i]: #neurons when they look different than before
                try:                                       #to advance frames faster
                    self.neur_circs[lay][i].undraw()
                except:
                    pass
                col = (255*np.asarray(cmap(val/disc)[0:3])).astype(int)
                self.neur_circs[lay][i] = g.Circle(g.Point(x, ys[i]), radius = 15)
                self.neur_circs[lay][i].setOutline(g.color_rgb(*col))
                self.neur_circs[lay][i].setFill(g.color_rgb(*col))
                self.neur_circs[lay][i].draw(self.win)
                self.neur_vals_disc[lay][i] = val
    
    def draw_network_neurons(self):
        game_vars = self.game_vars()
        neuron_layers = []
        for i in range(2):
            neuron_layer = np.squeeze(self.hidden_layers[i](np.asarray([[game_vars[0],game_vars[1]]])))
            neuron_layers += [neuron_layer]
        self.draw_neuron_help(np.asarray(game_vars), 0, colscale = 1, first = True)
        self.draw_neuron_help(neuron_layers[0], 1, colscale = 1)
        self.draw_neuron_help(neuron_layers[1], 2, colscale = 1)
        
    def step(self, robot = None):
        reward = 0
        scored_bool = False
        died_bool = False
        curr_game_vars = self.game_vars()
        self.v += Game.a
        if robot is not None:
            if np.argmax(robot(np.asarray(curr_game_vars).reshape(1,2))): #If the vector is [0 1] i.e. JUMP
                self.v = -200
                self.jumps += 1
        elif self.graph and self.win.checkKey() == 'space':
            self.v = -200  
        self.rec.move(0, self.v*Game.dt)
        for i,pipe in enumerate(self.pipes):
            pipe.move()
            if pipe.check_coll(self) or self.check_bounds():
                if self.graph:
                    g.update(self.frame_rate)
                    time.sleep(2)
                    self.win.close()
                self.over = True
                died_bool = True
                #print('dead')
            elif pipe.check_kill(self):
                self.pipes[i] = Game.Pipe(Game.pipe_dx, Game.pipe_space*3 - Game.pipe_dx, self.pipe_ran(), Game.pipe_v, self.pipe_dy, self.win)
                if self.graph:
                    self.pipes[i].draw(self)
                self.score += 1
                scored_bool = True           
        
        if self.graph and (self.success_model is not None):
            self.draw_network_neurons()
        
        #Cookie allocation
        if died_bool:
            reward = -1 
#        elif scored_bool:   #Reward for getting a point?
#            reward = 1
        else:
            tdist, ddist = curr_game_vars
            r2 = (tdist**2 + ddist**2)
           # reward = 0.01  + min(60**3/(r2**1.5), 1) #FOR SUCCESS 1-3
            #reward = np.exp(-0.5*r2/120**2) #Gaussian 
            reward = 0.01 + min(45**3/(r2**1.5), 1) #For final success model
        done = died_bool or (self.score >= Game.thresh)
        self.t += Game.dt
        self.frame += 1
        if self.graph:
            g.update(self.frame_rate) #20 slows it down, 60 is really fast
        return curr_game_vars, reward, done, (self.frame, self.score, self.jumps)
    def action_step(self, action):
        if action == 0:
            return self.step(lambda foo: [1, 0])
        elif action == 1:
            return self.step(lambda foo: [0, 1])
        else: 
            raise ValueError("action must be 0 or 1 for this game")

 #%% To do the training
if __name__ == "__main__":
    #pass
   main() 

#%% To watch an old model play
from keras.models import load_model

#There's success 1 which works okay for dy 250, success 3 which works great for dy 250
# And success final which was trained on dy 200, but is good up to ~dy 165
success_model = load_model('successes/success final.model')
mygame = Game(graph = True, success_model = success_model, pipe_dy = 165, frame_rate = 60)
mygame.run(robot = success_model.predict) 
 #%% To play the game yourself with spacebar
mygame = Game(graph = True, frame_rate = 50, pipe_dy = 165)
mygame.run()   

