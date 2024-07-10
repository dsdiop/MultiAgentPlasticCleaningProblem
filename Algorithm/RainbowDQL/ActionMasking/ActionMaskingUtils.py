import numpy as np
import torch
from torch.nn.functional import softmax
from copy import deepcopy

class SafeActionMasking:

	def __init__(self, action_space_dim: int, movement_length: float) -> None:
		""" Safe Action Masking """

		self.navigation_map = None
		self.position = None
		self.angle_set = np.linspace(0, 2 * np.pi, action_space_dim, endpoint=False)
		self.movement_length = movement_length

	def update_state(self, position: np.ndarray, new_navigation_map: np.ndarray = None):
		""" Update the navigation map """

		if new_navigation_map is not None:
			self.navigation_map = new_navigation_map

		""" Update the position """
		self.position = position

	def mask_action(self, q_values: np.ndarray = None):

		#if q_values is None:
			#""" Random selection """
		#	q_values = np.random.rand(n_actions)

		movements = np.asarray([np.round(np.array([self.movement_length * np.cos(angle), self.movement_length * np.sin(angle)])).astype(int) for angle in self.angle_set])
		next_positions = self.position + movements
		# if the size of q_values is different from the number of actions, we need to add a position to next_positions that is the current position
		if q_values.size == 9:	
			next_positions = np.vstack((next_positions, self.position))
		world_collisions = [(new_position[0] < 0) or (new_position[0] >= self.navigation_map.shape[0]) or (new_position[1] < 0) or (new_position[1] >= self.navigation_map.shape[1])
					for new_position in next_positions]

		action_mask = np.array([self.isnot_reachable(next_position) 
                          if not world_collisions[i] else True for i,next_position in enumerate(next_positions)]).astype(bool)

		q_values[action_mask] = -np.inf

		return q_values, np.argmax(q_values)

	def isnot_reachable(self, next_position):
		""" Check if the next position is reachable or navigable """
		if self.navigation_map[int(next_position[0]), int(next_position[1])] == 0:
			return True 
		x, y = next_position
		dx = x - self.position[0]
		dy = y - self.position[1]
		steps = max(abs(dx), abs(dy))
		dx = dx / steps if steps != 0 else 0
		dy = dy / steps if steps != 0 else 0
		reachable_positions = True
		for step in range(1, steps + 1):
			px = round(self.position[0] + dx * step)
			py = round(self.position[1] + dy * step)
			if self.navigation_map[px, py] != 1:
				reachable_positions = False
				break

		return not reachable_positions

class NoGoBackMasking:

	def __init__(self) -> None:
		
		self.previous_action = None

	def mask_action(self, q_values: np.ndarray = None):

		if q_values is None:
			""" Random selection """
			q_values = np.random.rand(8)

		if self.previous_action is None:
			self.previous_action = np.argmax(q_values)
		else:
			
			return_action = (self.previous_action + len(q_values) // 2) % len(q_values)
			q_values[return_action] = -np.inf

		return q_values, np.argmax(q_values)

	def update_last_action(self, last_action):

		self.previous_action = last_action
	
class ConsensusSafeActionMasking:
	""" The optimists decide first! """

	def __init__(self, navigation_map, action_space_dim: int, movement_length: float) -> None:
		
		self.movement_length = movement_length
		self.angle_set = np.linspace(0, 2 * np.pi, action_space_dim, endpoint=False)
		self.position = None
		self.fleet_map = np.zeros_like(navigation_map)


	def query_actions(self, q_values, positions: np.ndarray):

		# 1) The largest q-value agent decides first
		# 2) If there are multiple agents with the same q-value, the agent is selected randomly
		# 3) Then, compute the next position of the agent and update the fleet map
		# 4) The next agent is selected based on the updated fleet map, etc
		
		self.fleet_map = np.ones_like(self.fleet_map)
		q_values_array = np.array(list(q_values.values()))
		sorted_indices = np.argsort(q_values_array.max(axis=1))[::-1]
		agents_order = list(np.array(list(q_values.keys()))[sorted_indices])

		#agents_order = np.argsort(q_values.max(axis=1))[::-1]
		final_actions = {agent_id: 0 for agent_id in q_values.keys()}
		not_solved = True
		count = 0
		while not_solved:
			count = count + 1
			q_values_copy = deepcopy(q_values)
			self.fleet_map = np.ones_like(self.fleet_map)
			for idx,agent in enumerate(agents_order):
				# Unpack the agent position
				agent_position = positions[agent]
				# Compute the impossible actions
				movements = np.asarray([np.round(np.array([self.movement_length * np.cos(angle), self.movement_length * np.sin(angle)])) for angle in self.angle_set]).astype(int)
				#movements = np.asarray([np.round(np.array([np.cos(angle), np.sin(angle)])) * self.movement_length for angle in self.angle_set]).astype(int)
				next_positions = agent_position + movements
				if q_values[agent].size == 9:	
					next_positions = np.vstack((next_positions, agent_position))
				world_collisions = [(new_position[0] < 0) or (new_position[0] >= self.fleet_map.shape[0]) or (new_position[1] < 0) or (new_position[1] >= self.fleet_map.shape[1])
                      		for new_position in next_positions]
				action_mask = np.array([self.fleet_map[int(next_position[0]), int(next_position[1])] == 0  if not world_collisions[i] else True 
                            for i,next_position in enumerate(next_positions)]).astype(bool)
				# Censor the impossible actions in the Q-values
				q_values_copy[agent][action_mask] = -np.inf
				# If all the actions of the are impossible, the agent should make the first move
				if all(q_value == -np.inf for q_value in q_values_copy[agent]):
					agents_order = [agent] + agents_order[:idx] + agents_order[idx+1:]
					break
				# Select the action
				action = np.argmax(q_values_copy[agent])

				# Update the fleet map
				next_position = next_positions[action]
				self.fleet_map[int(next_position[0]), int(next_position[1])] = 0

				# Store the action
				final_actions[agent] = action.copy()
				if idx == len(agents_order) - 1:
					not_solved = False
			if not_solved:
				print(f"Changed Agents order")
				print(f"agents_order: {agents_order}")
				print(f"q_values: {q_values}")
				print(f"q_values_copy: {q_values_copy}")
				print(f"count: {count}, idx: {idx}")
				print(f"positions: {positions}")
				print(self.fleet_map)
			if count == 100:
				breakpoint()
		return {agent: final_actions[agent] for agent in q_values.keys()}


class ConsensusSafeActionDistributionMasking:
	""" The same as ConsensusSafeActionMasking, but the action is selected from the action distribution, conditiones on the action mask """

	def __init__(self, navigation_map, action_space_dim: int, movement_length: float) -> None:
		
		self.movement_length = movement_length
		self.angle_set = np.linspace(0, 2 * np.pi, action_space_dim, endpoint=False)
		self.position = None
		self.fleet_map = np.zeros_like(navigation_map)

	def query_actions_from_logits(self, logits: torch.Tensor, positions: np.ndarray, device, deterministic: bool = False):

		# 1) The largest q-value agent decides first
		# 2) If there are multiple agents with the same q-value, the agent is selected randomly
		# 3) Then, compute the next position of the agent and update the fleet map
		# 4) The next agent is selected based on the updated fleet map, etc
		
		self.fleet_map = np.ones_like(self.fleet_map)
		agents_order = np.argsort(logits.cpu().detach().numpy().max(axis=1))[::-1]
		final_actions = torch.zeros(logits.shape[0], dtype=int, device=device)
		action_log_probs = torch.zeros(logits.shape[0], dtype=float, device=device)
		entropy = torch.zeros(logits.shape[0], dtype=float, device=device)

		for agent in agents_order:
			
			#Unpack the agent position
			agent_position = positions[agent]
			# Compute the impossible actions
			movements = np.asarray([np.round(np.array([np.cos(angle), np.sin(angle)])) * self.movement_length for angle in self.angle_set]).astype(int)
			next_positions = agent_position + movements
			action_mask = np.array([self.fleet_map[int(next_position[0]), int(next_position[1])] == 0 for next_position in next_positions]).astype(bool)
			# Censor the impossible actions in the Q-values
			logits[agent][action_mask] = -torch.finfo(torch.float).max

			# Select the action
			action_probabilities = softmax(logits[agent], dim=0)
			action_distribution = torch.distributions.Categorical(probs=action_probabilities)
			if deterministic:
				action = action_distribution.mode
			else:
				action = action_distribution.sample()

			action_log_probs[agent] = action_distribution.log_prob(action)
			entropy[agent] = action_distribution.entropy().mean()
			

			# Update the fleet map
			next_position = next_positions[action]
			self.fleet_map[int(next_position[0]), int(next_position[1])] = 0

			# Store the action
			final_actions[agent] = action

		return final_actions, action_log_probs, entropy

		
		