import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import random
from collections import defaultdict
import time

# 1. Representação do Cubo 3D (2x2x2 com Blocos Nomeados)
class RubiksCube3D:
    def __init__(self, size=2):
        self.size = size
        self.faces = {
            'F': np.full((size, size), 'G'),  # Front (Green)
            'B': np.full((size, size), 'B'),  # Back (Blue)
            'U': np.full((size, size), 'W'),  # Up (White)
            'D': np.full((size, size), 'Y'),  # Down (Yellow)
            'L': np.full((size, size), 'O'),  # Left (Orange)
            'R': np.full((size, size), 'R')   # Right (Red)
        }
        self.rotation_angle = 0
        self.rotating_face = None
        self.rotation_direction = 1
        self.animation_speed = 8
        self.is_animating = False
        self.move_history = []
        # Definir os 8 blocos de canto com nomes e cores iniciais
        self.corners = {
            'C1': {'pos': (0, 1, 1), 'colors': {'U': 'W', 'F': 'G', 'L': 'O'}},  # Top-Front-Left
            'C2': {'pos': (1, 1, 1), 'colors': {'U': 'W', 'F': 'G', 'R': 'R'}},  # Top-Front-Right
            'C3': {'pos': (0, 1, 0), 'colors': {'U': 'W', 'B': 'B', 'L': 'O'}},  # Top-Back-Left
            'C4': {'pos': (1, 1, 0), 'colors': {'U': 'W', 'B': 'B', 'R': 'R'}},  # Top-Back-Right
            'C5': {'pos': (0, 0, 1), 'colors': {'D': 'Y', 'F': 'G', 'L': 'O'}},  # Bottom-Front-Left
            'C6': {'pos': (1, 0, 1), 'colors': {'D': 'Y', 'F': 'G', 'R': 'R'}},  # Bottom-Front-Right
            'C7': {'pos': (0, 0, 0), 'colors': {'D': 'Y', 'B': 'B', 'L': 'O'}},  # Bottom-Back-Left
            'C8': {'pos': (1, 0, 0), 'colors': {'D': 'Y', 'B': 'B', 'R': 'R'}},  # Bottom-Back-Right
        }

    def get_state(self):
        """Retorna o estado atual como uma string baseada nos blocos"""
        state_parts = []
        for corner_name in sorted(self.corners.keys()):
            corner = self.corners[corner_name]
            pos = corner['pos']
            colors = ''.join(corner['colors'][face] for face in sorted(corner['colors'].keys()))
            state_parts.append(f"{corner_name}:{pos[0]}{pos[1]}{pos[2]}:{colors}")
        return '|'.join(state_parts)

    def is_solved(self):
        """Verifica se cada face tem uma única cor"""
        for face_key, face in self.faces.items():
            if not np.all(face == face[0, 0]):
                return False
        face_colors = [self.faces[face_key][0, 0] for face_key in ['F', 'B', 'U', 'D', 'L', 'R']]
        return len(set(face_colors)) == 6

    def apply_move(self, move):
        """Aplica um movimento ao cubo (inicia animação)"""
        if self.is_animating:
            return False
        self.rotating_face, direction = move
        self.rotation_direction = 1 if direction == 'u' else -1
        self.rotation_angle = 0
        self.is_animating = True
        direction_symbol = "" if direction == 'u' else "'"
        self.move_history.append(f"{self.rotating_face}{direction_symbol}")
        return True

    def update_animation(self):
        """Atualiza a animação de rotação"""
        if not self.is_animating:
            return False
        self.rotation_angle += self.rotation_direction * self.animation_speed
        if abs(self.rotation_angle) >= 90:
            self.complete_rotation()
            self.is_animating = False
            return True
        return False

    def complete_rotation(self):
        """Completa a rotação da face"""
        face = self.rotating_face
        direction = 'u' if self.rotation_direction == 1 else 'd'
        face_data = self.faces[face]
        if direction == 'u':
            self.faces[face] = np.rot90(face_data, 3)
        else:
            self.faces[face] = np.rot90(face_data, 1)
        self._rotate_adjacent_edges_2x2(face, direction)
        self.rotation_angle = 0
        self.rotating_face = None

    def _rotate_adjacent_edges_2x2(self, face, direction):
        """Rotaciona as bordas adjacentes e atualiza blocos"""
        if face == 'F':
            U_edge = self.faces['U'][1, :].copy()
            R_edge = self.faces['R'][:, 0].copy()
            D_edge = self.faces['D'][0, :].copy()
            L_edge = self.faces['L'][:, 1].copy()
            if direction == 'u':
                self.faces['U'][1, :] = L_edge[::-1]
                self.faces['L'][:, 1] = D_edge
                self.faces['D'][0, :] = R_edge[::-1]
                self.faces['R'][:, 0] = U_edge
                self._update_corner('C1', (0, 0, 1), {'D': self.faces['D'][0, 0], 'F': self.faces['F'][1, 0], 'L': self.faces['L'][0, 1]})
                self._update_corner('C2', (0, 1, 1), {'U': self.faces['U'][1, 0], 'F': self.faces['F'][1, 1], 'L': self.faces['L'][1, 1]})
                self._update_corner('C5', (1, 0, 1), {'D': self.faces['D'][0, 1], 'F': self.faces['F'][0, 0], 'R': self.faces['R'][0, 0]})
                self._update_corner('C6', (1, 1, 1), {'U': self.faces['U'][1, 1], 'F': self.faces['F'][0, 1], 'R': self.faces['R'][1, 0]})
            else:
                self.faces['U'][1, :] = R_edge
                self.faces['R'][:, 0] = D_edge[::-1]
                self.faces['D'][0, :] = L_edge
                self.faces['L'][:, 1] = U_edge[::-1]
                self._update_corner('C1', (1, 1, 1), {'U': self.faces['U'][1, 1], 'F': self.faces['F'][0, 1], 'R': self.faces['R'][1, 0]})
                self._update_corner('C2', (0, 1, 1), {'U': self.faces['U'][1, 0], 'F': self.faces['F'][1, 1], 'L': self.faces['L'][1, 1]})
                self._update_corner('C5', (0, 0, 1), {'D': self.faces['D'][0, 0], 'F': self.faces['F'][1, 0], 'L': self.faces['L'][0, 1]})
                self._update_corner('C6', (1, 0, 1), {'D': self.faces['D'][0, 1], 'F': self.faces['F'][0, 0], 'R': self.faces['R'][0, 0]})
        elif face == 'B':
            U_edge = self.faces['U'][0, :].copy()
            L_edge = self.faces['L'][:, 0].copy()
            D_edge = self.faces['D'][1, :].copy()
            R_edge = self.faces['R'][:, 1].copy()
            if direction == 'u':
                self.faces['U'][0, :] = R_edge
                self.faces['R'][:, 1] = D_edge[::-1]
                self.faces['D'][1, :] = L_edge
                self.faces['L'][:, 0] = U_edge[::-1]
                self._update_corner('C3', (1, 1, 0), {'U': self.faces['U'][0, 1], 'B': self.faces['B'][0, 1], 'R': self.faces['R'][1, 1]})
                self._update_corner('C4', (0, 1, 0), {'U': self.faces['U'][0, 0], 'B': self.faces['B'][0, 0], 'L': self.faces['L'][1, 0]})
                self._update_corner('C7', (1, 0, 0), {'D': self.faces['D'][1, 1], 'B': self.faces['B'][1, 1], 'R': self.faces['R'][0, 1]})
                self._update_corner('C8', (0, 0, 0), {'D': self.faces['D'][1, 0], 'B': self.faces['B'][1, 0], 'L': self.faces['L'][0, 0]})
            else:
                self.faces['U'][0, :] = L_edge[::-1]
                self.faces['L'][:, 0] = D_edge
                self.faces['D'][1, :] = R_edge[::-1]
                self.faces['R'][:, 1] = U_edge
                self._update_corner('C3', (0, 1, 0), {'U': self.faces['U'][0, 0], 'B': self.faces['B'][0, 0], 'L': self.faces['L'][1, 0]})
                self._update_corner('C4', (1, 1, 0), {'U': self.faces['U'][0, 1], 'B': self.faces['B'][0, 1], 'R': self.faces['R'][1, 1]})
                self._update_corner('C7', (0, 0, 0), {'D': self.faces['D'][1, 0], 'B': self.faces['B'][1, 0], 'L': self.faces['L'][0, 0]})
                self._update_corner('C8', (1, 0, 0), {'D': self.faces['D'][1, 1], 'B': self.faces['B'][1, 1], 'R': self.faces['R'][0, 1]})
        elif face == 'U':
            B_edge = self.faces['B'][0, :].copy()
            R_edge = self.faces['R'][0, :].copy()
            F_edge = self.faces['F'][0, :].copy()
            L_edge = self.faces['L'][0, :].copy()
            if direction == 'u':
                self.faces['F'][0, :] = R_edge
                self.faces['R'][0, :] = B_edge
                self.faces['B'][0, :] = L_edge
                self.faces['L'][0, :] = F_edge
                self._update_corner('C1', (0, 1, 0), {'U': self.faces['U'][1, 0], 'B': self.faces['B'][0, 0], 'L': self.faces['L'][0, 0]})
                self._update_corner('C2', (1, 1, 0), {'U': self.faces['U'][1, 1], 'B': self.faces['B'][0, 1], 'R': self.faces['R'][0, 1]})
                self._update_corner('C3', (0, 1, 1), {'U': self.faces['U'][0, 0], 'F': self.faces['F'][0, 0], 'L': self.faces['L'][0, 1]})
                self._update_corner('C4', (1, 1, 1), {'U': self.faces['U'][0, 1], 'F': self.faces['F'][0, 1], 'R': self.faces['R'][0, 0]})
            else:
                self.faces['F'][0, :] = L_edge
                self.faces['L'][0, :] = B_edge
                self.faces['B'][0, :] = R_edge
                self.faces['R'][0, :] = F_edge
                self._update_corner('C1', (0, 1, 1), {'U': self.faces['U'][0, 0], 'F': self.faces['F'][0, 0], 'L': self.faces['L'][0, 1]})
                self._update_corner('C2', (1, 1, 1), {'U': self.faces['U'][0, 1], 'F': self.faces['F'][0, 1], 'R': self.faces['R'][0, 0]})
                self._update_corner('C3', (0, 1, 0), {'U': self.faces['U'][1, 0], 'B': self.faces['B'][0, 0], 'L': self.faces['L'][0, 0]})
                self._update_corner('C4', (1, 1, 0), {'U': self.faces['U'][1, 1], 'B': self.faces['B'][0, 1], 'R': self.faces['R'][0, 1]})
        elif face == 'D':
            F_edge = self.faces['F'][1, :].copy()
            R_edge = self.faces['R'][1, :].copy()
            B_edge = self.faces['B'][1, :].copy()
            L_edge = self.faces['L'][1, :].copy()
            if direction == 'u':
                self.faces['F'][1, :] = L_edge
                self.faces['L'][1, :] = B_edge
                self.faces['B'][1, :] = R_edge
                self.faces['R'][1, :] = F_edge
                self._update_corner('C5', (0, 0, 0), {'D': self.faces['D'][1, 0], 'B': self.faces['B'][1, 0], 'L': self.faces['L'][1, 0]})
                self._update_corner('C6', (1, 0, 0), {'D': self.faces['D'][1, 1], 'B': self.faces['B'][1, 1], 'R': self.faces['R'][1, 1]})
                self._update_corner('C7', (0, 0, 1), {'D': self.faces['D'][0, 0], 'F': self.faces['F'][1, 0], 'L': self.faces['L'][1, 1]})
                self._update_corner('C8', (1, 0, 1), {'D': self.faces['D'][0, 1], 'F': self.faces['F'][1, 1], 'R': self.faces['R'][1, 0]})
            else:
                self.faces['F'][1, :] = R_edge
                self.faces['R'][1, :] = B_edge
                self.faces['B'][1, :] = L_edge
                self.faces['L'][1, :] = F_edge
                self._update_corner('C5', (0, 0, 1), {'D': self.faces['D'][0, 0], 'F': self.faces['F'][1, 0], 'L': self.faces['L'][1, 1]})
                self._update_corner('C6', (1, 0, 1), {'D': self.faces['D'][0, 1], 'F': self.faces['F'][1, 1], 'R': self.faces['R'][1, 0]})
                self._update_corner('C7', (0, 0, 0), {'D': self.faces['D'][1, 0], 'B': self.faces['B'][1, 0], 'L': self.faces['L'][1, 0]})
                self._update_corner('C8', (1, 0, 0), {'D': self.faces['D'][1, 1], 'B': self.faces['B'][1, 1], 'R': self.faces['R'][1, 1]})
        elif face == 'L':
            U_edge = self.faces['U'][:, 0].copy()
            F_edge = self.faces['F'][:, 0].copy()
            D_edge = self.faces['D'][:, 0].copy()
            B_edge = self.faces['B'][:, 1].copy()
            if direction == 'u':
                self.faces['F'][:, 0] = D_edge
                self.faces['D'][:, 0] = B_edge[::-1]
                self.faces['B'][:, 1] = U_edge[::-1]
                self.faces['U'][:, 0] = F_edge
                self._update_corner('C1', (0, 0, 1), {'D': self.faces['D'][0, 0], 'F': self.faces['F'][1, 0], 'L': self.faces['L'][0, 1]})
                self._update_corner('C3', (0, 0, 0), {'D': self.faces['D'][1, 0], 'B': self.faces['B'][1, 1], 'L': self.faces['L'][0, 0]})
                self._update_corner('C5', (0, 1, 1), {'U': self.faces['U'][0, 0], 'F': self.faces['F'][0, 0], 'L': self.faces['L'][1, 1]})
                self._update_corner('C7', (0, 1, 0), {'U': self.faces['U'][1, 0], 'B': self.faces['B'][0, 1], 'L': self.faces['L'][1, 0]})
            else:
                self.faces['F'][:, 0] = U_edge
                self.faces['U'][:, 0] = B_edge[::-1]
                self.faces['B'][:, 1] = D_edge[::-1]
                self.faces['D'][:, 0] = F_edge
                self._update_corner('C1', (0, 1, 1), {'U': self.faces['U'][0, 0], 'F': self.faces['F'][0, 0], 'L': self.faces['L'][1, 1]})
                self._update_corner('C3', (0, 1, 0), {'U': self.faces['U'][1, 0], 'B': self.faces['B'][0, 1], 'L': self.faces['L'][1, 0]})
                self._update_corner('C5', (0, 0, 1), {'D': self.faces['D'][0, 0], 'F': self.faces['F'][1, 0], 'L': self.faces['L'][0, 1]})
                self._update_corner('C7', (0, 0, 0), {'D': self.faces['D'][1, 0], 'B': self.faces['B'][1, 1], 'L': self.faces['L'][0, 0]})
        elif face == 'R':
            U_edge = self.faces['U'][:, 1].copy()
            B_edge = self.faces['B'][:, 0].copy()
            D_edge = self.faces['D'][:, 1].copy()
            F_edge = self.faces['F'][:, 1].copy()
            if direction == 'u':
                self.faces['F'][:, 1] = U_edge
                self.faces['U'][:, 1] = B_edge[::-1]
                self.faces['B'][:, 0] = D_edge[::-1]
                self.faces['D'][:, 1] = F_edge
                self._update_corner('C2', (1, 0, 1), {'D': self.faces['D'][0, 1], 'F': self.faces['F'][0, 1], 'R': self.faces['R'][0, 0]})
                self._update_corner('C4', (1, 0, 0), {'D': self.faces['D'][1, 1], 'B': self.faces['B'][1, 0], 'R': self.faces['R'][0, 1]})
                self._update_corner('C6', (1, 1, 1), {'U': self.faces['U'][0, 1], 'F': self.faces['F'][1, 1], 'R': self.faces['R'][1, 0]})
                self._update_corner('C8', (1, 1, 0), {'U': self.faces['U'][1, 1], 'B': self.faces['B'][0, 0], 'R': self.faces['R'][1, 1]})
            else:
                self.faces['F'][:, 1] = D_edge
                self.faces['D'][:, 1] = B_edge[::-1]
                self.faces['B'][:, 0] = U_edge[::-1]
                self.faces['U'][:, 1] = F_edge
                self._update_corner('C2', (1, 1, 1), {'U': self.faces['U'][0, 1], 'F': self.faces['F'][1, 1], 'R': self.faces['R'][1, 0]})
                self._update_corner('C4', (1, 1, 0), {'U': self.faces['U'][1, 1], 'B': self.faces['B'][0, 0], 'R': self.faces['R'][1, 1]})
                self._update_corner('C6', (1, 0, 1), {'D': self.faces['D'][0, 1], 'F': self.faces['F'][0, 1], 'R': self.faces['R'][0, 0]})
                self._update_corner('C8', (1, 0, 0), {'D': self.faces['D'][1, 1], 'B': self.faces['B'][1, 0], 'R': self.faces['R'][0, 1]})

    def _update_corner(self, corner_name, new_pos, new_colors):
        """Atualiza a posição e cores de um bloco de canto"""
        self.corners[corner_name]['pos'] = new_pos
        self.corners[corner_name]['colors'] = new_colors

    def scramble(self, n=8):
        """Embaralha o cubo com n movimentos aleatórios"""
        self.move_history = []
        moves = self.get_possible_moves()
        for _ in range(n):
            move = random.choice(moves)
            self._apply_move_instantly(move)
            direction_symbol = "" if move[1] == 'u' else "'"
            self.move_history.append(f"{move[0]}{direction_symbol}")

    def _apply_move_instantly(self, move):
        """Aplica um movimento instantaneamente"""
        face, direction = move
        face_data = self.faces[face]
        if direction == 'u':
            self.faces[face] = np.rot90(face_data, 3)
        else:
            self.faces[face] = np.rot90(face_data, 1)
        self._rotate_adjacent_edges_2x2(face, direction)

    def get_possible_moves(self):
        """Retorna todos os movimentos possíveis"""
        faces = ['F', 'B', 'U', 'D', 'L', 'R']
        directions = ['u', 'd']  # u: clockwise, d: counterclockwise
        return [(face, direction) for face in faces for direction in directions]

    def count_solved_faces(self):
        """Conta quantas faces têm uma única cor"""
        return sum(1 for face in self.faces.values() if np.all(face == face[0, 0]))

    def print_moves(self):
        """Imprime os movimentos realizados"""
        if not self.move_history:
            print("Nenhum movimento realizado.")
            return
        print("\nHistórico de Movimentos:")
        print(" ".join(self.move_history))
        print(f"Total de movimentos: {len(self.move_history)}")

    def print_corners(self):
        """Imprime os blocos de canto com posições e cores"""
        print("\nEstado dos Blocos de Canto:")
        for corner_name, corner in sorted(self.corners.items()):
            pos = corner['pos']
            colors = corner['colors']
            print(f"{corner_name}: Pos({pos[0]}, {pos[1]}, {pos[2]}), Colors: {colors}")
        print("\nCores das Faces:")
        for face_key in ['F', 'B', 'U', 'D', 'L', 'R']:
            face = self.faces[face_key]
            color = face[0, 0]
            is_uniform = np.all(face == color)
            print(f"Face {face_key}: Cor {color}, Uniforme: {is_uniform}")

# 2. Ambiente RL
class RubiksEnv:
    def __init__(self):
        self.cube = RubiksCube3D()
        self.action_space = self.cube.get_possible_moves()
        self.max_steps = 200
        self.moves_count = 0
        self.last_action = None
        self.reset()

    def reset(self):
        """Reseta o ambiente e embaralha o cubo"""
        self.cube = RubiksCube3D()
        self.cube.scramble(6)
        self.moves_count = 0
        self.last_action = None
        return self.cube.get_state()

    def step(self, action):
        """Executa uma ação"""
        old_solved_faces = self.cube.count_solved_faces()
        self.cube._apply_move_instantly(action)
        self.moves_count += 1
        new_state = self.cube.get_state()
        new_solved_faces = self.cube.count_solved_faces()
        reward = self._calculate_reward(old_solved_faces, new_solved_faces)

        if self.last_action == action:
            reward -= 50
        self.last_action = action

        done = self.cube.is_solved() or self.moves_count >= self.max_steps
        return new_state, reward, done

    def _calculate_reward(self, old_solved_faces, new_solved_faces):
        """Calcula a recompensa"""
        if self.cube.is_solved():
            return 10000

        improvement = new_solved_faces - old_solved_faces
        face_reward = new_solved_faces * 500
        improvement_reward = improvement * 2000
        move_penalty = -10

        if new_solved_faces >= 5:
            face_reward += 3000
        elif new_solved_faces >= 4:
            face_reward += 2000
        elif new_solved_faces >= 3:
            face_reward += 1000
        elif new_solved_faces >= 2:
            face_reward += 500

        return face_reward + improvement_reward + move_penalty

# 3. Agente Q-Learning
class QLearningAgent:
    def __init__(self, env, max_episodes=10000, learning_rate=0.1, discount_factor=0.99,
                 exploration_rate=1.0, exploration_decay=0.999, min_exploration=0.01):
        self.env = env
        self.q_table = defaultdict(lambda: np.zeros(len(env.action_space)))
        self.max_episodes = max_episodes
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration = min_exploration
        self.episodes_completed = 0
        self.solved_count = 0
        self.training_stopped = False
        self.rewards_history = []
        self.steps_history = []

    def get_action(self, state):
        """Escolhe uma ação usando ε-greedy"""
        if random.random() < self.exploration_rate:
            return random.choice(range(len(self.env.action_space)))
        else:
            q_values = self.q_table[state]
            q_values_with_noise = q_values + np.random.normal(0, 0.01, len(q_values))
            return np.argmax(q_values_with_noise)

    def update(self, state, action, reward, next_state):
        """Atualiza a Q-table"""
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error

    def decay_exploration(self):
        """Reduz a taxa de exploração"""
        self.exploration_rate = max(self.min_exploration, self.exploration_rate * self.exploration_decay)

    def train(self, max_steps=200, verbose=False):
        """Treina o agente"""
        print("\n=== Iniciando Treinamento ===")
        print(f"Parâmetros: α={self.learning_rate:.2f}, γ={self.discount_factor:.2f}, ε={self.exploration_rate:.2f}")

        for episode in range(1, self.max_episodes + 1):
            state = self.env.reset()
            total_reward = 0
            steps = 0

            if verbose or episode % 100 == 0:
                print(f"\nEpisódio {episode}/{self.max_episodes}, ε={self.exploration_rate:.3f}")

            while steps < max_steps and not self.env.cube.is_solved():
                action_idx = self.get_action(state)
                action = self.env.action_space[action_idx]
                next_state, reward, done = self.env.step(action)
                self.update(state, action_idx, reward, next_state)
                total_reward += reward
                steps += 1
                state = next_state

                if verbose or episode % 100 == 0:
                    direction_symbol = "" if action[1] == 'u' else "'"
                    print(f"{steps}. Move: {action[0]}{direction_symbol} | Reward: {reward:.2f} | "
                          f"Faces Completas: {self.env.cube.count_solved_faces()}/6")

            self.rewards_history.append(total_reward)
            self.steps_history.append(steps)

            if self.env.cube.is_solved():
                self.solved_count += 1
                if verbose or episode % 100 == 0:
                    print(f"✅ Resolvido em {steps} movimentos! Recompensa: {total_reward:.2f}")
            else:
                if verbose or episode % 100 == 0:
                    print(f"❌ Não resolvido em {steps} movimentos. Recompensa: {total_reward:.2f}")

            self.episodes_completed += 1
            if self.episodes_completed >= self.max_episodes:
                self.training_stopped = True
                print(f"Treinamento concluído após {self.max_episodes} episódios")

            self.decay_exploration()

        success_rate = self.solved_count / self.max_episodes * 100
        print("\n=== Treinamento Concluído ===")
        print(f"Sucesso: {self.solved_count}/{self.max_episodes} ({success_rate:.1f}%)")
        if self.steps_history:
            last_100 = self.steps_history[-100:] if len(self.steps_history) >= 100 else self.steps_history
            print(f"Média de movimentos (últimos {len(last_100)} episódios): {sum(last_100)/len(last_100):.1f}")

        return self.rewards_history, self.steps_history

    def test_solution(self, gui=None, max_steps=100):
        """Testa a solução aprendida"""
        print("\n=== Testando Política ===")
        state = self.env.reset()
        total_reward = 0
        steps = 0
        plan = []
        visited_states = set()

        print(f"Estado inicial: {self.env.cube.count_solved_faces()}/6 faces completas")
        self.env.cube.print_corners()
        self.env.cube.move_history = []

        while steps < max_steps and not self.env.cube.is_solved():
            if state in visited_states:
                print("⚠️ Loop detectado. Parando...")
                break
            visited_states.add(state)

            action_idx = self.get_action(state)
            action = self.env.action_space[action_idx]
            if gui:
                gui.cube.apply_move(action)
                while gui.cube.is_animating:
                    gui.cube.update_animation()
                    gui.draw_cube()
                    pygame.display.flip()
                    pygame.time.wait(16)
            else:
                self.env.cube._apply_move_instantly(action)
            direction_symbol = "" if action[1] == 'u' else "'"
            self.env.cube.move_history.append(f"{action[0]}{direction_symbol}")

            next_state, reward, done = self.env.step(action)
            total_reward += reward
            steps += 1
            plan.append((action, reward, next_state))

            print(f"{steps}. Move: {action[0]}{direction_symbol} | Reward: {reward:.2f} | "
                  f"Faces Completas: {self.env.cube.count_solved_faces()}/6")

            state = next_state
            if done and self.env.cube.is_solved():
                print(f"✅ Resolvido em {steps} movimentos! Recompensa: {total_reward:.2f}")
                break
            elif steps == max_steps:
                print(f"❌ Não resolvido em {max_steps} movimentos. Recompensa: {total_reward:.2f}")

        self.env.cube.print_moves()
        self.env.cube.print_corners()
        return plan, total_reward

# 4. Interface Gráfica 3D
class Rubiks3DGUI:
    def __init__(self, agent):
        pygame.init()
        self.width, self.height = 800, 600
        self.screen = pygame.display.set_mode((self.width, self.height), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Q-Learning Cubo Mágico 2x2x2")

        self.agent = agent
        self.cube = agent.env.cube

        glViewport(0, 0, self.width, self.height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, (self.width / self.height), 0.1, 50.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, -6)
        glEnable(GL_DEPTH_TEST)
        glClearColor(0.1, 0.1, 0.2, 1.0)

        self.colors = {
            'W': (1, 1, 1), 'Y': (1, 1, 0), 'G': (0, 0.8, 0),
            'B': (0, 0, 1), 'O': (1, 0.5, 0), 'R': (1, 0, 0)
        }

        self.rotation_x = 20
        self.rotation_y = -30
        self.last_pos = None
        self.training = False
        self.running = True

    def draw_cube(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, -6)
        glRotatef(self.rotation_x, 1, 0, 0)
        glRotatef(self.rotation_y, 0, 1, 0)

        cube_size = 1.5
        piece_size = cube_size / 2

        for x in range(2):
            for y in range(2):
                for z in range(2):
                    glPushMatrix()
                    glTranslatef((x - 0.5) * piece_size, (y - 0.5) * piece_size, (z - 0.5) * piece_size)
                    if self.cube.is_animating and self.is_piece_in_rotating_face(x, y, z):
                        self.apply_rotation_transform()
                    self.draw_piece(x, y, z)
                    glPopMatrix()

    def is_piece_in_rotating_face(self, x, y, z):
        if not self.cube.rotating_face:
            return False
        face = self.cube.rotating_face
        if face == 'F' and z == 1:
            return True
        if face == 'B' and z == 0:
            return True
        if face == 'U' and y == 1:
            return True
        if face == 'D' and y == 0:
            return True
        if face == 'L' and x == 0:
            return True
        if face == 'R' and x == 1:
            return True
        return False

    def apply_rotation_transform(self):
        face = self.cube.rotating_face
        angle = self.cube.rotation_angle
        if face in ['F', 'B']:
            glRotatef(angle if face == 'F' else -angle, 0, 0, 1)
        elif face in ['U', 'D']:
            glRotatef(angle if face == 'U' else -angle, 0, 1, 0)
        elif face in ['L', 'R']:
            glRotatef(angle if face == 'R' else -angle, 1, 0, 0)

    def draw_piece(self, x, y, z):
        piece_size = 0.75
        front_color = self.cube.faces['F'][y, x] if z == 1 else None
        back_color = self.cube.faces['B'][1 - y, x] if z == 0 else None
        up_color = self.cube.faces['U'][1 - x, z] if y == 1 else None
        down_color = self.cube.faces['D'][x, z] if y == 0 else None
        left_color = self.cube.faces['L'][1 - y, z] if x == 0 else None
        right_color = self.cube.faces['R'][y, z] if x == 1 else None

        glBegin(GL_QUADS)
        if front_color:
            glColor3fv(self.colors[front_color])
            glVertex3f(-0.5 * piece_size, -0.5 * piece_size, 0.5 * piece_size)
            glVertex3f(0.5 * piece_size, -0.5 * piece_size, 0.5 * piece_size)
            glVertex3f(0.5 * piece_size, 0.5 * piece_size, 0.5 * piece_size)
            glVertex3f(-0.5 * piece_size, 0.5 * piece_size, 0.5 * piece_size)
        if back_color:
            glColor3fv(self.colors[back_color])
            glVertex3f(-0.5 * piece_size, -0.5 * piece_size, -0.5 * piece_size)
            glVertex3f(-0.5 * piece_size, 0.5 * piece_size, -0.5 * piece_size)
            glVertex3f(0.5 * piece_size, 0.5 * piece_size, -0.5 * piece_size)
            glVertex3f(0.5 * piece_size, -0.5 * piece_size, -0.5 * piece_size)
        if up_color:
            glColor3fv(self.colors[up_color])
            glVertex3f(-0.5 * piece_size, 0.5 * piece_size, -0.5 * piece_size)
            glVertex3f(-0.5 * piece_size, 0.5 * piece_size, 0.5 * piece_size)
            glVertex3f(0.5 * piece_size, 0.5 * piece_size, 0.5 * piece_size)
            glVertex3f(0.5 * piece_size, 0.5 * piece_size, -0.5 * piece_size)
        if down_color:
            glColor3fv(self.colors[down_color])
            glVertex3f(-0.5 * piece_size, -0.5 * piece_size, -0.5 * piece_size)
            glVertex3f(0.5 * piece_size, -0.5 * piece_size, -0.5 * piece_size)
            glVertex3f(0.5 * piece_size, -0.5 * piece_size, 0.5 * piece_size)
            glVertex3f(-0.5 * piece_size, -0.5 * piece_size, 0.5 * piece_size)
        if left_color:
            glColor3fv(self.colors[left_color])
            glVertex3f(-0.5 * piece_size, -0.5 * piece_size, -0.5 * piece_size)
            glVertex3f(-0.5 * piece_size, -0.5 * piece_size, 0.5 * piece_size)
            glVertex3f(-0.5 * piece_size, 0.5 * piece_size, 0.5 * piece_size)
            glVertex3f(-0.5 * piece_size, 0.5 * piece_size, -0.5 * piece_size)
        if right_color:
            glColor3fv(self.colors[right_color])
            glVertex3f(0.5 * piece_size, -0.5 * piece_size, -0.5 * piece_size)
            glVertex3f(0.5 * piece_size, 0.5 * piece_size, -0.5 * piece_size)
            glVertex3f(0.5 * piece_size, 0.5 * piece_size, 0.5 * piece_size)
            glVertex3f(0.5 * piece_size, -0.5 * piece_size, 0.5 * piece_size)
        glEnd()

        glColor3f(0, 0, 0)
        glBegin(GL_LINES)
        for edge in [
            ((-0.5 * piece_size, -0.5 * piece_size, 0.5 * piece_size), (0.5 * piece_size, -0.5 * piece_size, 0.5 * piece_size)),
            ((0.5 * piece_size, -0.5 * piece_size, 0.5 * piece_size), (0.5 * piece_size, 0.5 * piece_size, 0.5 * piece_size)),
            ((0.5 * piece_size, 0.5 * piece_size, 0.5 * piece_size), (-0.5 * piece_size, 0.5 * piece_size, 0.5 * piece_size)),
            ((-0.5 * piece_size, 0.5 * piece_size, 0.5 * piece_size), (-0.5 * piece_size, -0.5 * piece_size, 0.5 * piece_size)),
            ((-0.5 * piece_size, -0.5 * piece_size, -0.5 * piece_size), (0.5 * piece_size, -0.5 * piece_size, -0.5 * piece_size)),
            ((0.5 * piece_size, -0.5 * piece_size, -0.5 * piece_size), (0.5 * piece_size, 0.5 * piece_size, -0.5 * piece_size)),
            ((0.5 * piece_size, 0.5 * piece_size, -0.5 * piece_size), (-0.5 * piece_size, 0.5 * piece_size, -0.5 * piece_size)),
            ((-0.5 * piece_size, 0.5 * piece_size, -0.5 * piece_size), (-0.5 * piece_size, -0.5 * piece_size, -0.5 * piece_size)),
            ((-0.5 * piece_size, -0.5 * piece_size, -0.5 * piece_size), (-0.5 * piece_size, -0.5 * piece_size, 0.5 * piece_size)),
            ((0.5 * piece_size, -0.5 * piece_size, -0.5 * piece_size), (0.5 * piece_size, -0.5 * piece_size, 0.5 * piece_size)),
            ((0.5 * piece_size, 0.5 * piece_size, -0.5 * piece_size), (0.5 * piece_size, 0.5 * piece_size, 0.5 * piece_size)),
            ((-0.5 * piece_size, 0.5 * piece_size, -0.5 * piece_size), (-0.5 * piece_size, 0.5 * piece_size, 0.5 * piece_size)),
        ]:
            glVertex3f(*edge[0])
            glVertex3f(*edge[1])
        glEnd()

    def test_solution(self):
        """Testa a solução aprendida"""
        print("🧪 Testando solução aprendida")
        self.training = False
        self.agent.training_stopped = True
        self.cube.move_history = []
        self.agent.env.reset()
        plan, total_reward = self.agent.test_solution(gui=self)
        print(f"✅ Teste concluído com {len(plan)} movimentos, Recompensa: {total_reward:.2f}")
        if self.cube.is_solved():
            print("🎉 Todas as faces têm uma única cor!")
        else:
            print(f"⚠️ Cubo não resolvido, {self.cube.count_solved_faces()}/6 faces completas")

    def run(self):
        """Loop principal"""
        clock = pygame.time.Clock()
        trained = False

        while self.running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    self.running = False
                elif event.type == KEYDOWN:
                    if event.key == K_SPACE and not trained:
                        print("Iniciando treinamento...")
                        self.agent.train(max_steps=200, verbose=True)
                        print("Treinamento concluído. Pressione T para testar.")
                        trained = True
                        self.training = False
                        self.agent.training_stopped = True
                    elif event.key == K_t:
                        self.test_solution()
                    elif event.key == K_r:
                        self.agent.env.reset()
                        print("🔄 Cubo resetado")
                    elif event.key == K_s:
                        self.cube.scramble(8)
                        print("🎲 Cubo embaralhado")
                    elif event.key == K_m:
                        self.cube.print_moves()
                    elif event.key == K_c:
                        self.cube.print_corners()
                    elif event.key == K_q:
                        self.running = False
                    elif event.key == K_UP:
                        self.rotation_x = max(-90, self.rotation_x - 5)
                    elif event.key == K_DOWN:
                        self.rotation_x = min(90, self.rotation_x + 5)
                    elif event.key == K_LEFT:
                        self.rotation_y -= 5
                    elif event.key == K_RIGHT:
                        self.rotation_y += 5
                elif event.type == MOUSEBUTTONDOWN:
                    if event.button == 4:
                        glTranslatef(0, 0, 0.3)
                    elif event.button == 5:
                        glTranslatef(0, 0, -0.3)
                elif event.type == MOUSEMOTION:
                    if event.buttons[0]:
                        if self.last_pos:
                            dx = event.pos[0] - self.last_pos[0]
                            dy = event.pos[1] - self.last_pos[1]
                            self.rotation_y += dx * 0.5
                            self.rotation_x += dy * 0.5
                            self.rotation_x = max(-90, min(90, self.rotation_x))
                        self.last_pos = event.pos
                    else:
                        self.last_pos = None

            if self.cube.is_animating:
                self.cube.update_animation()

            self.draw_cube()
            pygame.display.flip()
            clock.tick(60)

        pygame.quit()

# Programa Principal
if __name__ == "__main__":
    try:
        env = RubiksEnv()
        agent = QLearningAgent(
            env,
            max_episodes=10,
            learning_rate=0.1,
            discount_factor=0.99,
            exploration_rate=1.0,
            exploration_decay=0.999,
            min_exploration=0.01
        )
        gui = Rubiks3DGUI(agent)
        gui.run()
    except Exception as e:
        print(f"Erro: {e}")
        pygame.quit()
        raise