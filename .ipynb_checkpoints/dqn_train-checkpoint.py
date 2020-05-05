import os
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from q_train import QNTrain


class DQNTrain(QNTrain):
    """
    Class for training a DQN
    """
    def __init__(
        self,
        q_net_class,
        env,
        config,
        device,
        logger=None,
    ):
        super().__init__(env, config, logger)
        self.device = device

        self.q_net = q_net_class(env, config)
        self.target_q_net = q_net_class(env, config)

        self.q_net.to(device)
        self.target_q_net.to(device)
        
        self.gamma = config.gamma

        self.update_target_params()
        for param in self.target_q_net.parameters():
            param.requires_grad = False
        self.target_q_net.eval()

        if config.optim_type == 'adam':
            self.optimizer = optim.Adam(
                self.q_net.parameters(), lr=config.lr_begin)
        elif config.optim_type == 'rmsprop':
            self.optimizer = optim.RMSprop(
                self.q_net.parameters(), lr=config.lr_begin)
        else:
            raise ValueError(f"Unknown optim_type: {config.optim_type}")


    def process_state(self, state):
        """
        Processing of state

        Args:
            state: np.ndarray of shape either (batch_size, H, W, C)
            or (H, W, C), of dtype 'np.uint8'

        Returns:
            state: A torch float tensor on self.device of shape
            (*, H, W, C), where * = batch_size if it was present in
            input, 1 otherwise. State is normalized by dividing by
            self.config.high
        """
        #####################################################################
        # TODO: Process state to match the return output specified above.
        #####################################################################
        state = torch.Tensor(state)
        if len(state.shape) == 3:
            desired_shape = (1, *state.shape)
            state = state.view(*desired_shape)
        #####################################################################
        #                             END OF YOUR CODE                      #
        #####################################################################
        return state


    def forward_loss(
        self,
        state,
        action,
        reward,
        next_state,
        done_mask,
    ):
        """
        Compute loss for a batch of transitions. Transitions are defined as
        tuples of (state, action, reward, next_state, done).

        Args:
            state: batch of states (batch_size, *)
            action: batch of actions (batch_size, num_actions)
            next_state: batch of next states (batch_size, *)
            reward: batch of rewards (batch_size)
            done_mask: batch of boolean values, 1 if next state is terminal
                state (ending the episode) and 0 otherwise.

        Returns:
            The loss for a transition is defined as:
                Q_samp(s) = r if done
                          = r + gamma * max_a' Q_target(s', a')
                loss = (Q_samp(s) - Q(s, a))^2

            Notation:
                s, s': current and next state respectively
                a: current action
                a': possible future actions at s'
                Q: self.q_net
                Q_target: self.target_q_net
        """
        #####################################################################
        # TODO: Compute the loss defined above by using the Q networks
        #   self.q_net and self.target_q_net
        #
        # HINT: The output of the Q network is a tensor of state-action
        #       values for ALL actions, but Q(s, a) denotes the state-action
        #       values for ONLY the actions taken by the agent. With the
        #       help of torch.gather, Q(s, a) can be computed given the
        #       batch of current states and actions.
        #
        #       After computing the Q(s, a) tensor of shape (batch_size),
        #       use the done_mask to compute Q_samp(s) from the equation
        #       specified above.
        #####################################################################
        '''
        Loss for each transition is the squared error between the Q-value of 
        our network for the chosen action and the Q-value observed from
        interaction.
        Q-value observed from interaction is just the reward if the next state
        is terminal. If the next state is not terminal, Q-value observed from
        interaction is the reward + the discounted future rewards. Future 
        rewards are given by the target network for a state and a list of
        possible actions. The maximum reward for all actions is then discounted
        and added to the current reward to give the Q-value observed from
        interaction.
        '''
        # Find the Q value from our network for all samples
        pred_q_matrix = self.q_net(state) # batch_size x num_actions
        pred_q_values = pred_q_matrix.gather(1, action.view(-1,1)).squeeze()

        # Find the Q values from experience
        #  Initially set to reward
        exp_q_values = torch.zeros_like(pred_q_values)
        exp_q_values += reward
                
        # For nonterminal next states, add future reward as well
        #  Get nonterminal next states 
        nonterminal_next_states = next_state[(1-done_mask).bool()]

        #  Get Q-values for nonterminal next states
        nonterm_q_matrix = self.target_q_net(nonterminal_next_states)
        
        #  Get best Q-value for each nonterminal next state
        nonterm_q_values = nonterm_q_matrix.max(1)[0]
    
        #  Add discounted future rewards for nonterminal next states
        exp_q_values[(1-done_mask).bool()] += self.gamma*nonterm_q_values

        # Compute loss
        loss = torch.sum((exp_q_values - pred_q_values)**2)
        #####################################################################
        #                             END OF YOUR CODE                      #
        #####################################################################
        return loss


    def update_target_params(self):
        """
        Update parametes of Q' with parameters of Q
        """
        #####################################################################
        # TODO: Update the parameters of self.target_q_net with the
        # parameters of self.q_net, refer to the documentation of
        # torch.nn.Module.load_state_dict and torch.nn.Module.state_dict
        # This should just take 1-2 lines of code.
        #####################################################################
        q_net_params = self.q_net.state_dict()
        self.target_q_net.load_state_dict(q_net_params)
        for param in self.target_q_net.parameters():
            param.requires_grad = False
        #####################################################################
        #                             END OF YOUR CODE                      #
        #####################################################################


    def module_grad_norm(self, net):
        """
        Compute the L2 norm of gradients accumulated in net
        """
        with torch.no_grad():
            total_norm = 0
            for param in net.parameters():
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = np.sqrt(total_norm)
            return total_norm


    def save(self):
        """
        Saves session
        """
        if not os.path.exists(self.config.model_output):
            os.makedirs(self.config.model_output)

        torch.save(self.q_net.state_dict(),
            os.path.join(self.config.model_output,
                f"{self.q_net.__class__.__name__}.vd"))


    def get_best_action(self, state):
        """
        Return best action

        Args:
            state: 4 consecutive observations from gym
        Returns:
            action: (int)
            action_values: (np array) q values for all actions
        """
        state = self.process_state(state)
        action_values = self.q_net(state)

        return np.argmax(action_values.cpu().numpy()), action_values


    def update_step(self, t, replay_buffer, lr):
        """
        Performs an update of parameters by sampling from replay_buffer

        Args:
            t: number of iteration (episode and move)
            replay_buffer: ReplayBuffer instance .sample() gives batches
            lr: (float) learning rate

        Returns:
            q_loss: Loss computed using self.forward_loss
            grad_norm_eval: L2 norm of self.q_net gradients, computed
                using self.module_grad_norm
        """

        s_batch, a_batch, r_batch, sp_batch, done_mask_batch \
            = replay_buffer.sample(self.config.batch_size)

        #####################################################################
        # TODOs:
        # 1. Process all the arrays sampled from the replay buffer
        #   states -> use self.process_state
        #   everything else -> convert np.ndarrays to torch tensors
        #   and move them to self.device
        #   NOTE: actions must be converted to long tensors using .long()
        #
        # 2. After you implement self.forward_loss, use it here with
        #   the processed tensors as input to compute q_loss.
        #
        # 3. Update the Q network with one step of self.optimizer after
        #   calling backward on q_loss. Make sure to compute the L2 norm
        #   of the gradients using self.module_grad_norm on self.q_net
        #   AFTER calling backward.
        #####################################################################
        # Process arrays
        s_batch = self.process_state(s_batch)
        sp_batch = self.process_state(sp_batch)
        a_batch = torch.Tensor(a_batch).long().to(self.device)
        r_batch = torch.Tensor(r_batch).to(self.device)
        done_mask_batch = torch.Tensor(done_mask_batch).to(self.device)

        self.optimizer.zero_grad()
        q_loss = self.forward_loss(s_batch, a_batch, r_batch, sp_batch, done_mask_batch)
        q_loss.backward()
        grad_norm_eval = self.module_grad_norm(self.q_net)
        self.optimizer.step()
        #####################################################################
        #                             END OF YOUR CODE                      #
        #####################################################################
        return q_loss, grad_norm_eval
