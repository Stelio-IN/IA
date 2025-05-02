import tkinter as tk
from tkinter import ttk, messagebox
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
from collections import defaultdict, deque

# Environment setup
positions = ["canto1", "canto2", "canto3", "canto4", "centro"]

# Distances between positions
distances = {
    ("canto1", "centro"): 3.535, ("centro", "canto1"): 3.535,
    ("canto2", "centro"): 3.535, ("centro", "canto2"): 3.535,
    ("canto3", "centro"): 3.535, ("centro", "canto3"): 3.535,
    ("canto4", "centro"): 3.535, ("centro", "canto4"): 3.535,
    ("canto1", "canto2"): 5, ("canto2", "canto1"): 5,
    ("canto2", "canto3"): 5, ("canto3", "canto2"): 5,
    ("canto3", "canto4"): 5, ("canto4", "canto3"): 5,
    ("canto4", "canto1"): 5, ("canto1", "canto4"): 5,
    ("canto1", "canto3"): 7.07, ("canto3", "canto1"): 7.07,
    ("canto2", "canto4"): 7.07, ("canto4", "canto2"): 7.07,
}

# Enhanced action costs
ACTION_COSTS = {
    "mover": 1.0,
    "pegar_vara": 2.0,
    "colocar_vara": 1.5,
    "empurrar_cadeira": 4.0,
    "subir_cadeira": 3.0,
    "apontar_vara": 2.5,
    "agitar_vara": 2.0,
    "pular": 5.0,
    "arremessar_vara": 10.0
}


def dist(a, b):
    """Calculate distance between two positions"""
    if a == b:
        return 0
    return distances.get((a, b), 100)


class Estado:
    def __init__(self, pos_macaco, pos_cadeira, pos_vara, pos_bananas, tem_vara, em_cima, tem_bananas,
                 vara_na_cadeira=False, vara_apontada=False):
        self.pos_macaco = pos_macaco
        self.pos_cadeira = pos_cadeira
        self.pos_vara = pos_vara
        self.pos_bananas = pos_bananas
        self.tem_vara = tem_vara
        self.em_cima = em_cima
        self.tem_bananas = tem_bananas
        self.vara_na_cadeira = vara_na_cadeira
        self.vara_apontada = vara_apontada

    def __eq__(self, other):
        return isinstance(other, Estado) and self.__dict__ == other.__dict__

    def __hash__(self):
        return hash(tuple(self.__dict__.values()))

    def __repr__(self):
        return str(self.__dict__)

    def to_tuple(self):
        """Convert state to tuple for Q-table key"""
        return (
            self.pos_macaco,
            self.pos_cadeira,
            self.pos_vara,
            self.pos_bananas,
            self.tem_vara,
            self.em_cima,
            self.tem_bananas,
            self.vara_na_cadeira,
            self.vara_apontada
        )


def objetivo(estado):
    """Check if goal is reached (monkey has bananas)"""
    return estado.tem_bananas


def obter_acoes_possiveis(estado):
    """Returns all possible actions with (action_name, new_state, reward)"""
    acoes_possiveis = []

    # Basic movement between positions
    for p in positions:
        if p != estado.pos_macaco:
            novo = Estado(p, estado.pos_cadeira, estado.pos_vara, estado.pos_bananas,
                          estado.tem_vara, False, estado.tem_bananas,
                          estado.vara_na_cadeira, estado.vara_apontada)
            cost = dist(estado.pos_macaco, p) * ACTION_COSTS["mover"]
            acoes_possiveis.append(("mover_para_" + p, novo, -cost))

    # Chair-related actions
    if estado.pos_macaco == estado.pos_cadeira:
        if not estado.em_cima:
            # Move chair to new position
            for p in positions:
                if p != estado.pos_macaco:
                    novo = Estado(p, p, estado.pos_vara, estado.pos_bananas,
                                  estado.tem_vara, False, estado.tem_bananas,
                                  estado.vara_na_cadeira, estado.vara_apontada)
                    cost = dist(estado.pos_macaco, p) * ACTION_COSTS["empurrar_cadeira"]
                    acoes_possiveis.append(("mover_a_cadeira_para_" + p, novo, -cost))

            # Chair climbing actions
            acoes_possiveis.append(("subir_na_cadeira",
                                    Estado(estado.pos_macaco, estado.pos_cadeira, estado.pos_vara,
                                           estado.pos_bananas, estado.tem_vara, True, estado.tem_bananas,
                                           estado.vara_na_cadeira, estado.vara_apontada),
                                    -ACTION_COSTS["subir_cadeira"]))

            if estado.tem_vara:
                acoes_possiveis.append(("subir_na_cadeira_com_vara",
                                        Estado(estado.pos_macaco, estado.pos_cadeira, estado.pos_vara,
                                               estado.pos_bananas, True, True, estado.tem_bananas,
                                               False, estado.vara_apontada),
                                        -ACTION_COSTS["subir_cadeira"]))

        else:  # On chair
            acoes_possiveis.append(("descer_da_cadeira",
                                    Estado(estado.pos_macaco, estado.pos_cadeira, estado.pos_vara,
                                           estado.pos_bananas, estado.tem_vara, False, estado.tem_bananas,
                                           estado.vara_na_cadeira, estado.vara_apontada),
                                    -ACTION_COSTS["subir_cadeira"]))

            if estado.tem_vara:
                acoes_possiveis.append(("descer_da_cadeira_com_vara",
                                        Estado(estado.pos_macaco, estado.pos_cadeira, estado.pos_vara,
                                               estado.pos_bananas, True, False, estado.tem_bananas,
                                               False, estado.vara_apontada),
                                        -ACTION_COSTS["subir_cadeira"]))

    # Stick-related actions
    if estado.pos_macaco == estado.pos_vara and not estado.tem_vara and not estado.em_cima:
        acoes_possiveis.append(("pegar_a_vara",
                                Estado(estado.pos_macaco, estado.pos_cadeira, estado.pos_macaco,
                                       estado.pos_bananas, True, estado.em_cima, estado.tem_bananas,
                                       False, estado.vara_apontada),
                                -ACTION_COSTS["pegar_vara"]))

    if estado.tem_vara and not estado.em_cima:
        acoes_possiveis.append(("largar_a_vara_no_chao",
                                Estado(estado.pos_macaco, estado.pos_cadeira, estado.pos_macaco,
                                       estado.pos_bananas, False, estado.em_cima, estado.tem_bananas,
                                       False, estado.vara_apontada),
                                -ACTION_COSTS["colocar_vara"]))

    if estado.tem_vara and estado.pos_macaco == estado.pos_cadeira and not estado.em_cima:
        acoes_possiveis.append(("colocar_a_vara_sobre_cadeira",
                                Estado(estado.pos_macaco, estado.pos_cadeira, estado.pos_cadeira,
                                       estado.pos_bananas, False, estado.em_cima, estado.tem_bananas,
                                       True, estado.vara_apontada),
                                -ACTION_COSTS["colocar_vara"]))

    if estado.vara_na_cadeira and estado.pos_macaco == estado.pos_cadeira and estado.em_cima:
        acoes_possiveis.append(("remover_a_vara_sobre_cadeira",
                                Estado(estado.pos_macaco, estado.pos_cadeira, estado.pos_cadeira,
                                       estado.pos_bananas, True, True, estado.tem_bananas,
                                       False, estado.vara_apontada),
                                -ACTION_COSTS["pegar_vara"]))

    # Banana-related actions
    if estado.pos_macaco == estado.pos_bananas:
        if estado.em_cima and estado.tem_vara and not estado.vara_apontada:
            acoes_possiveis.append(("apontar_a_vara_para_as_bananas",
                                    Estado(estado.pos_macaco, estado.pos_cadeira, estado.pos_vara,
                                           estado.pos_bananas, True, True, estado.tem_bananas,
                                           estado.vara_na_cadeira, True),
                                    -ACTION_COSTS["apontar_vara"]))

        if estado.vara_apontada and estado.tem_vara and estado.em_cima:
            acoes_possiveis.append(("agitar_a_vara",
                                    Estado(estado.pos_macaco, estado.pos_cadeira, estado.pos_vara,
                                           estado.pos_bananas, True, True, True,
                                           estado.vara_na_cadeira, True),
                                    100 - ACTION_COSTS["agitar_vara"]))

        # Jumping actions (usually ineffective)
        if not estado.em_cima:
            acoes_possiveis.append(("pular",
                                    Estado(estado.pos_macaco, estado.pos_cadeira, estado.pos_vara,
                                           estado.pos_bananas, estado.tem_vara, False, False,
                                           estado.vara_na_cadeira, estado.vara_apontada),
                                    -ACTION_COSTS["pular"]))

            acoes_possiveis.append(("saltar_em_pe",
                                    Estado(estado.pos_macaco, estado.pos_cadeira, estado.pos_vara,
                                           estado.pos_bananas, estado.tem_vara, False, False,
                                           estado.vara_na_cadeira, estado.vara_apontada),
                                    -ACTION_COSTS["pular"]))

            if estado.tem_vara:
                acoes_possiveis.append(("saltar_com_a_vara_na_mao",
                                        Estado(estado.pos_macaco, estado.pos_cadeira, estado.pos_vara,
                                               estado.pos_bananas, True, False, False,
                                               estado.vara_na_cadeira, estado.vara_apontada),
                                        -ACTION_COSTS["pular"]))

                acoes_possiveis.append(("arremessar_a_vara",
                                        Estado(estado.pos_macaco, estado.pos_cadeira, None,
                                               estado.pos_bananas, False, False, False,
                                               False, False),
                                        -ACTION_COSTS["arremessar_vara"]))

        if estado.pos_macaco == estado.pos_cadeira and estado.em_cima:
            acoes_possiveis.append(("pular_sobre_a_cadeira",
                                    Estado(estado.pos_macaco, estado.pos_cadeira, estado.pos_vara,
                                           estado.pos_bananas, estado.tem_vara, False, False,
                                           estado.vara_na_cadeira, estado.vara_apontada),
                                    -ACTION_COSTS["pular"] * 2))  # More dangerous action

    return acoes_possiveis


class SmartMonkeyQLearning:
    def __init__(self, estado_inicial):
        self.q_table = defaultdict(float)
        self.failed_attempts = defaultdict(int)
        self.ineffective_actions = set()
        self.visited_states = set()
        self.repeated_actions = defaultdict(int)
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 0.8
        self.min_exploration = 0.05
        self.exploration_decay = 0.997
        self.estado_inicial = estado_inicial
        self.rewards_history = []
        self.steps_history = []
        self.useless_actions_log = []

    def get_action(self, estado):
        state_key = estado.to_tuple()
        possible_actions = obter_acoes_possiveis(estado)

        # Filter out known useless actions and recently repeated actions
        valid_actions = []
        for action in possible_actions:
            action_name = action[0]
            action_state_key = (state_key, action_name)

            if (action_state_key in self.ineffective_actions or
                    self.repeated_actions.get(action_state_key, 0) > 2):
                self.useless_actions_log.append(f"Avoided {action_name} at {state_key}")
                continue

            valid_actions.append(action)

        if not valid_actions:
            return None

        # Exploration
        if random.random() < self.exploration_rate:
            chosen_action = random.choice(valid_actions)
        else:
            # Exploitation with penalty for repeated actions
            q_values = []
            for action in valid_actions:
                action_name = action[0]
                action_state_key = (state_key, action_name)
                base_q = self.q_table.get(action_state_key, 0)
                repetition_penalty = self.repeated_actions.get(action_state_key, 0) * -5
                q_values.append(base_q + repetition_penalty)

            max_q = max(q_values)
            best_actions = [a for a, q in zip(valid_actions, q_values) if q == max_q]
            chosen_action = random.choice(best_actions)

        # Track this action
        action_state_key = (state_key, chosen_action[0])
        self.repeated_actions[action_state_key] += 1

        return chosen_action

    def learn_from_episode(self, episode_history):
        visited_in_episode = set()

        for i, (state, action, reward, next_state) in enumerate(episode_history):
            state_key = state.to_tuple()
            action_name = action[0]
            action_state_key = (state_key, action_name)
            next_state_key = next_state.to_tuple()

            # Detect loops and useless actions
            if next_state_key in visited_in_episode:
                self.q_table[action_state_key] -= 10
                self.failed_attempts[action_state_key] += 1

            visited_in_episode.add(next_state_key)

            # Detect useless actions
            if reward < -5:
                self.failed_attempts[action_state_key] += 1

                if self.failed_attempts[action_state_key] > 2:
                    self.ineffective_actions.add(action_state_key)
                    self.q_table[action_state_key] = -100
                    self.useless_actions_log.append(
                        f"Marked {action_name} at {state_key} as useless")

            # Reward progress toward goal
            progress = dist(state.pos_macaco, state.pos_bananas) - dist(next_state.pos_macaco, next_state.pos_bananas)
            if progress > 0:
                self.q_table[action_state_key] += progress * 2

            # Q-learning update
            next_actions = obter_acoes_possiveis(next_state)
            if next_actions:
                best_next = max([self.q_table.get((next_state_key, a[0]), 0) for a in next_actions])
            else:
                best_next = 0

            current_q = self.q_table.get(action_state_key, 0)
            new_q = current_q + self.learning_rate * (reward + self.discount_factor * best_next - current_q)
            self.q_table[action_state_key] = new_q

    def train_episode(self, max_steps=100):
        estado = Estado(*self.estado_inicial.to_tuple())
        episode_history = []
        total_reward = 0
        visited_states = set()

        for step in range(max_steps):
            state_key = estado.to_tuple()

            if state_key in visited_states:
                total_reward -= 5

            visited_states.add(state_key)

            action = self.get_action(estado)
            if not action:
                break

            action_name, next_state, reward = action
            episode_history.append((estado, action, reward, next_state))

            estado = next_state
            total_reward += reward

            if objetivo(estado):
                total_reward += 100
                break

        self.learn_from_episode(episode_history)
        self.exploration_rate = max(self.min_exploration,
                                    self.exploration_rate * self.exploration_decay)

        return total_reward, step + 1

    def train(self, episodes=1000, verbose=True):
        self.rewards_history = []
        self.steps_history = []

        for episode in range(1, episodes + 1):
            reward, steps = self.train_episode()
            self.rewards_history.append(reward)
            self.steps_history.append(steps)

            if verbose and episode % 50 == 0:
                print(f"\nEpisódio {episode}:")
                print(f"Recompensa: {reward:.1f} em {steps} passos")
                print(f"Taxa exploração: {self.exploration_rate:.3f}")
                print(f"Ações inúteis conhecidas: {len(self.ineffective_actions)}")
                if self.useless_actions_log:
                    print("Últimas ações evitadas:")
                    for log in self.useless_actions_log[-3:]:
                        print("  " + log)
                self.useless_actions_log = []

        return self.rewards_history, self.steps_history

    def get_optimal_policy(self, estado_inicial, max_steps=50):
        estado = Estado(*estado_inicial.to_tuple())
        policy = []
        visited_states = defaultdict(int)
        last_states = deque(maxlen=3)  # Track recent states to detect small loops

        for step in range(max_steps):
            state_key = estado.to_tuple()
            visited_states[state_key] += 1
            last_states.append(state_key)

            # Check for small loops (repeating last 3 states)
            if len(last_states) == 3 and last_states[0] == last_states[2]:
                policy.append((step + 1, "LOOP DETECTADO - TENTANDO AÇÃO ALTERNATIVA", str(estado)))
                # Don't break, just mark that we're trying something different

            if objetivo(estado):
                policy.append((step + 1, "OBJETIVO ALCANÇADO!", str(estado)))
                break

            possible_actions = obter_acoes_possiveis(estado)
            if not possible_actions:
                policy.append((step + 1, "SEM AÇÕES POSSÍVEIS", str(estado)))
                break

            # Score all possible actions considering various factors
            scored_actions = []
            for action in possible_actions:
                action_name, next_state, reward = action
                action_state_key = (state_key, action_name)
                next_state_key = next_state.to_tuple()

                base_q = self.q_table.get(action_state_key, 0)
                # Penalize actions that lead to frequently visited states
                visited_penalty = visited_states.get(next_state_key, 0) * -5
                # Penalize repeated actions
                repetition_penalty = self.repeated_actions.get(action_state_key, 0) * -3
                # Bonus for progress toward bananas
                progress_bonus = (dist(estado.pos_macaco, estado.pos_bananas) -
                                  dist(next_state.pos_macaco, next_state.pos_bananas)) * 2

                total_score = base_q + visited_penalty + repetition_penalty + progress_bonus
                scored_actions.append((total_score, action))

            # Sort by best score first
            scored_actions.sort(reverse=True, key=lambda x: x[0])

            # Try to find an action that doesn't lead to a frequently visited state
            action_taken = None
            for score, action in scored_actions:
                _, next_state, _ = action
                next_state_key = next_state.to_tuple()

                if visited_states.get(next_state_key, 0) < 2:  # Only allow visiting a state twice
                    action_taken = action
                    break

            # If all actions lead to visited states, take the best