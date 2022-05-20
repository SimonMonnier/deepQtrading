import random
from re import S
from gym import Env
import math
import torch
from src.model import DQN
from src.memory import ReplayMemory, Experience
# from model import DQN
# from memory import ReplayMemory, Experience
from collections import namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
from IPython.display import clear_output
import wandb
import numpy as np

Config = namedtuple(
    'Config', ('target_update', 'lr', 'lr_min', 'lr_decay', 'gamma', 'loss', 'memory_size', 'batch_size', 'eps_start', 'eps_min', 'eps_decay'))

BUY = 0
SELL = 1
HOLD = 2

class DQNAgentOpen():
    def __init__(self, env=None, config: Config = {}, id="", input_shape=104):
        self.env = env

        
        # self.num_actions = self.env.action_space.n
        self.num_actions = 3
        
        self.policy_net = DQN(input_shape,
                              self.num_actions).to("cuda:0")
        self.target_net = DQN(input_shape,
                              self.num_actions).to("cuda:0")
        self.policy_net = self.policy_net.double()
        self.target_net = self.target_net.double()

        self.policy_net
        self.target_net

        if config:
            self.config = config
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()  # target network in eval mode, not training
            self.target_update = config.target_update
            self.lr = config.lr
            self.lr_min = config.lr_min
            self.lr_decay = config.lr_decay
            self.current_lr = self.lr
            self.gamma = config.gamma
            self.loss_type = config.loss
            self.optimizer = torch.optim.Adam(
                params=self.policy_net.parameters(), lr=self.lr)  # TODO try other optimizer

            self.memory = ReplayMemory(config.memory_size)
            self.batch_size = config.batch_size

            self.eps_start = config.eps_start
            self.eps_min = config.eps_min
            self.eps_decay = config.eps_decay

        self.device = torch.device("cuda:0")

        if id != "":
            self.id = str(id) + "-" + str(int(time.time()))
        else:
            self.id = str(int(time.time()))

    #
    # Get the value of x at current_step using logarithme decay
    #
    @staticmethod
    def log_decay(v_start, v_min, max_step, current_step):
        rate = math.log(v_min / v_start) * -1 / max_step
        value = v_min + (v_start * (1 - rate)**current_step)
        return value

    #
    # Get epsilon used for greedy strategy,
    # depending on current episode to apply decay
    #
    def _get_epsilon(self, episode):
        return self.log_decay(self.eps_start, self.eps_min, self.eps_decay, episode)

    #
    # Update the learning rate using logarithme decay
    #
    def _update_learning_rate(self, episode):
        
        
        self.current_lr = self.log_decay(
        self.lr, self.lr_min, self.lr_decay, episode)

        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.current_lr

    #
    # Choose an action depending on the state using policy network
    #
    def _choose_action(self, state):
        state = torch.tensor(np.array([state]), device=self.device)
        state = torch.flatten(state)
        with torch.no_grad():  # No grad because we use the model to select an action and not for training
            predicted = self.policy_net(state)
            return predicted.argmax().item()  # select max value index

    #
    # Select action during training,
    # choosing beetwen explore or exploit depending of epsilon
    #
    def _select_action(self, state, episode):
        epsilon = self._get_epsilon(episode)
        if epsilon > random.uniform(0, 1):
            return random.randrange(self.num_actions)  # explore
        else:
            return self._choose_action(state)  # exploit

    #
    # Memorize env state in agent memory
    #
    def _memorize(self, state, action, new_state, reward, done):
        state = np.array(state)
        new_state = np.array(new_state)
        experience = Experience(torch.tensor(np.array([state]), device=self.device),
                                torch.tensor([action], device=self.device),
                                torch.tensor(np.array([new_state]),
                                             device=self.device),
                                torch.tensor([reward], device=self.device),
                                torch.tensor([done], device=self.device, dtype=torch.bool))
        self.memory.push(experience)

    @staticmethod
    def extract_tensors(experiences):
        batch = Experience(*zip(*experiences))
        tensor_state = torch.cat(batch.state)
        tensor_action = torch.cat(batch.action)
        tensor_reward = torch.cat(batch.reward)
        tensor_next_state = torch.cat(batch.next_state)
        tensor_done = torch.cat(batch.done)
        return (tensor_state, tensor_action, tensor_reward, tensor_next_state, tensor_done)

    #
    # Train model with a sample of replay memory
    #
    def _train_model(self):
        experiences = self.memory.sample(self.batch_size)
        states_b, actions_b, rewards_b, new_states_b, _ = self.extract_tensors(
            experiences)
        states_b = torch.flatten(states_b, start_dim=1)
        new_states_b = torch.flatten(new_states_b, start_dim=1)
        current_q_values = self.policy_net(states_b).gather(dim=1, index=actions_b.unsqueeze(
            1))  # send all states and actions pairs and get list of predicted qvalues
        next_q_values = self.target_net(new_states_b).max(dim=1)[
            0]  # get target qvalues
        target_q_values = (next_q_values * self.gamma) + \
            rewards_b  # compute expected qvalues

        if self.loss_type == 'mse':
            loss = nn.functional.mse_loss(
                current_q_values, target_q_values.unsqueeze(1))
        elif self.loss_type == 'huber':
            loss = nn.functional.huber_loss(
                current_q_values, target_q_values.unsqueeze(1))
        elif self.loss_type == 'mae':
            loss = nn.functional.smooth_l1_loss(
                current_q_values, target_q_values.unsqueeze(1))

        self.optimizer.zero_grad()  # prevent accumulating gradients during backprop
        loss.backward()  # compute the gradient of the loss with respect of weight and biases in the policy net
        for param in self.policy_net.parameters():
            # clips gradients computed during backpropagation to avoid explosion of gradients
            param.grad.data.clamp_(-1, 1)
        # update the weight and biases with the just previous gradients computed
        self.optimizer.step()

    #
    # Function to train the model with
    # previously given parameters
    #
    def fit(self, wandb_log):
        if wandb_log == True:
            run = wandb.init(project="DeepTrading", entity="smonnier")
        num_episodes = 3000
        max_step = 756
        solde = 1000
        benefice = 0
        try:
            for episode in range(num_episodes):
                print("Training episode: {0}/{1}".format(episode+1, num_episodes), end="\r")

                state = self.env.reset()
                episode_reward = 0
                episode_action = [0,0,0]
                for step in range(max_step):
                    
                    action = self._select_action(state, episode)
                    episode_action[action] += 1
                    new_state, reward, done, info = self.env.step(action)
                    self._memorize(state, action, new_state, reward, done)
                    state = new_state
                    episode_reward += reward
                    

                    if self.memory.get_current_len() >= self.batch_size:  # Train model
                        self._train_model()

                    if done:
                        break

                self._update_learning_rate(episode)
                if episode % self.target_update == 0:
                    self.target_net.load_state_dict(
                        self.policy_net.state_dict())  # update target network

                # benefice = benefice + self.env.sold - solde

                if wandb_log == True:

                    wandb.log({"reward": episode_reward, "duration": step,
                              "epsilon": self._get_epsilon(episode), "learning_rate": self.current_lr,
                            #    "solde": self.env.sold, "total_benefice": benefice, 
                               "Buy": episode_action[BUY],
                            #    "Buy_2": episode_action[BUY_2],
                               "Sell": episode_action[SELL],
                            #    "Sell_2": episode_action[SELL_2],
                               "Hold": episode_action[HOLD]})
                            #    "Close": episode_action[CLOSE]
                            #    "Close_2": episode_action[CLOSE_2],
                            #    "Close_3": episode_action[CLOSE_3],
                            #     "Trade Sold": self.env.trade_sold,
                            #    "Total trade": self.env.total_trade})
            if wandb_log == True:
                run.finish()
            self._save()  # save trained model
        except KeyboardInterrupt:
            self._save()
            if wandb_log == True:
                run.finish()
            print("Training has been interrupted")

    #
    # Save model weight into file
    #
    def _save(self):
        with open('./src/models/{0}.pt'.format(self.id), 'w') as f:
            torch.save(self.policy_net.state_dict(),
                       f"./src/models/{self.id}.pt")
            print('Model saved as: {0}.pt'.format(self.id))
        with open('./src/models/{0}.config.txt'.format(self.id), 'w') as f:
            t, p = self.evaluate(100)
            f.write(str(self.config).replace(',', ',\n'))
            f.write('\n---------------------------\n')
            f.write('Results after 100 episodes:\n')
            f.write(f"Average timesteps per episode: {t}\n")
            f.write(f"Average penalties per episode: {p}")

    def load(self, file_path):
        self.policy_net.load_state_dict(torch.load(file_path))
        self.policy_net.eval()  # model in eval mode, not training

    def play_game(self, n_game):
        max_step = 1440
        for episode in range(n_game):
            state = self.env.reset()
            game = "*** Game {0} ***\n".format(episode + 1)
            episode_reward = 0
            time.sleep(1)
            for _ in range(max_step):
                clear_output(wait=True)
                print(game, self.env.render('human'), end='\n')
                time.sleep(0.3)
                action = self._choose_action(state)
                state, reward, done = self.env.step(action)
                episode_reward += reward
                if done:
                    clear_output(wait=True)
                    print(game, self.env.render('human'), end='\n')
                    if reward == 1:
                        print("*** You Won {0} ***".format(episode_reward))
                        time.sleep(2)
                    else:
                        print("*** Your score {0} ***".format(episode_reward))
                        time.sleep(2)
                    clear_output(wait=True)
                    break

    def evaluate(self, wandb_log):
        if wandb_log == True:
            run = wandb.init(project="DeepTrading", entity="smonnier")
        num_episodes = 2000
        max_step = 2880
        solde = 1000
        benefice = 0
        try:
            for episode in range(num_episodes):
                print(
                    "Training episode: {0}/{1}".format(episode+1, num_episodes), end="\r")

                state = self.env.reset()
                episode_reward = 0
                episode_action = [0,0,0,0,0,0,0,0]
                for step in range(max_step):
                    
                    action = self._choose_action(state)
                    episode_action[action] += 1
                    new_state, reward, done, info = self.env.step(action)
                    state = new_state
                    episode_reward += reward

                    if done:
                        break

                

                if wandb_log == True:

                    wandb.log({"reward": episode_reward, "duration": step,
                              "epsilon": self._get_epsilon(episode), "learning_rate": self.current_lr,
                               "Hold": episode_action[HOLD],
                               "Close": episode_action[CLOSE]})
            if wandb_log == True:
                run.finish()
            self._save()  # save trained model
        except KeyboardInterrupt:
            self._save()
            if wandb_log == True:
                run.finish()
            print("Training has been interrupted")