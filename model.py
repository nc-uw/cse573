from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.misc_util import norm_col_init, weights_init
import torchvision

class ModelInput:
    """ Input to the model. """
    def __init__(self, state=None, hidden=None, additional_state_info = None):
        self.state = state
        self.hidden = hidden
	# initialization for additional_state_info
	self.additional_state_info = additional_state_info

class ModelOutput:
    """ Output of the model. """
    def __init__(
            self,
            value=None,
            policy=None,
            hidden=None):

        self.value = value
        self.policy = policy
        self.hidden = hidden

class Model(torch.nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, stride=1, padding=2)
        self.maxp1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, 5, stride=1, padding=1)
        self.maxp2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 4, stride=1, padding=1)
        self.maxp3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.maxp4 = nn.MaxPool2d(2, 2)
        
        # augState encoding
        additional_state_size = 2
        augmented_hidden_size = 10
        self.augmented_linear = nn.Linear(additional_state_size, augmented_hidden_size)
	self.augmented_combination = nn.Linear(1024 + augmented_hidden_size, 1024)
        
        self.lstm = nn.LSTMCell(1024, args.hidden_state_sz)
        self.critic_linear = nn.Linear(args.hidden_state_sz, 1)
        self.actor_linear = nn.Linear(args.hidden_state_sz, args.action_space)

        self.apply(weights_init)
        relu_gain = nn.init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv3.weight.data.mul_(relu_gain)
        self.conv4.weight.data.mul_(relu_gain)
	#augState weights
	self.augmented_linear.weight.data = norm_col_init(
            self.augmented_linear.weight.data, 0.01)
        self.augmented_linear.bias.data.fill_(0)
        self.augmented_combination.weight.data = norm_col_init(
            self.augmented_combination.weight.data, 0.01)
        self.augmented_combination.bias.data.fill_(0)
	
        self.actor_linear.weight.data = norm_col_init(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = norm_col_init(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.train()

    def embedding(self, state, additional_state_info):
        x = F.relu(self.maxp1(self.conv1(state)))
        x = F.relu(self.maxp2(self.conv2(x)))
        x = F.relu(self.maxp3(self.conv3(x)))
        x = F.relu(self.maxp4(self.conv4(x)))

        x = x.view(x.size(0), -1)
        
        # augState embedding
	additional_score = self.augmented_linear(additional_state_info)
	augmented_x = self.augmented_combination(torch.cat([x, additional_score], dim=1))
	return augmented_x
	#return x

    def a3clstm(self, x, hidden):
        hx, cx = self.lstm(x, hidden)
        x = hx
        critic_out = self.critic_linear(x)
        actor_out = self.actor_linear(x)
        return actor_out, critic_out, (hx, cx)

    def forward(self, model_input):
        state = model_input.state
	additional_state_info = model_input.additional_state_info #augState
	#check-print
	#print ('forward model.py ..printing additional_state_info tensor..', additional_state_info.data[0])
	
	(hx, cx) = model_input.hidden
	x = self.embedding(state, additional_state_info) #pass augState to embedding
	#x = self.embedding(state)
	actor_out, critic_out, (hx, cx) = self.a3clstm(x, (hx, cx))
	
	return ModelOutput(policy=actor_out, value=critic_out, hidden=(hx, cx))
