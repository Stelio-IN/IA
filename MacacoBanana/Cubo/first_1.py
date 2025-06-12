import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import random
from collections import defaultdict
import time
import math
import sys


# 1. Representação do Cubo 3D (Versão Simplificada 2x2x2)
class RubiksCube3D:
    def __init__(self, size=2):
        self.size = size
        self.faces = {
            'F': np.full((size, size), 'G'),  # Frente (Green)
            'B': np.full((size, size), 'B'),  # Trás (Blue)
            'U': np.full((size, size), 'W'),  # Cima (White)
            'D': np.full((size, size), 'Y'),  # Baixo (Yellow)
            'L': np.full((size, size), 'O'),  # Esquerda (Orange)
            'R': np.full((size, size), 'R')  # Direita (Red)
        }
        self.rotation_angle = 0
        self.rotating_face = None
        self.rotation_direction = 1
        self.animation_speed = 8
        self.is_animating = False
        self.move_history = []

    def get_state(self):
        """Retorna o estado atual como uma string compacta"""
        state_str = ""
        for face_key in ['F', 'B', 'U', 'D', 'L', 'R']:
            face = self.faces[face_key]
            state_str += ''.join(face.flatten())
        return state_str

    def is_solved(self):
        """Verifica se o cubo está resolvido"""
        # Primeiro, verifica se cada face é de uma única cor
        for face in self.faces.values():
            if not np.all(face == face[0, 0]):
                return False
        # Depois, verifica se cada face tem uma cor diferente
        face_colors = [self.faces[face_key][0, 0] for face_key in ['F', 'B', 'U', 'D', 'L', 'R']]
        if len(set(face_colors)) != 6:  # Deve haver 6 cores diferentes
            return False
        return True

    def apply_move(self, move):
        """Aplica um movimento ao cubo (inicia animação)"""
        if self.is_animating:
            return False

        self.rotating_face, direction = move
        self.rotation_direction = 1 if direction == 'clockwise' else -1
        self.rotation_angle = 0
        self.is_animating = True

        direction_symbol = "" if direction == 'clockwise' else "'"
        self.move_history.append(f"{self.rotating_face}{direction_symbol}")

        # Debug: Imprime o estado do cubo antes do movimento
        print(f"\nAntes do movimento {self.rotating_face}{direction_symbol}:")
        for face_key, face_data in self.faces.items():
            print(f"{face_key}: {face_data}")

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
        """Completa a rotação da face (atualiza o estado do cubo)"""
        face = self.rotating_face
        direction = 'clockwise' if self.rotation_direction == 1 else 'counterclockwise'

        face_data = self.faces[face]
        if direction == 'clockwise':
            self.faces[face] = np.rot90(face_data, 3)
        else:
            self.faces[face] = np.rot90(face_data, 1)

        self._rotate_adjacent_edges_2x2(face, direction)

        self.rotation_angle = 0
        self.rotating_face = None

        # Debug: Imprime o estado do cubo após o movimento
        print(f"Após o movimento {face}{'clockwise' if direction == 'clockwise' else 'counterclockwise'}:")
        for face_key, face_data in self.faces.items():
            print(f"{face_key}: {face_data}")

    def _rotate_adjacent_edges_2x2(self, face, direction):
        """Rotaciona as bordas adjacentes para cubo 2x2 de forma explícita"""
        if face == 'F':
            U_edge = self.faces['U'][1, :].copy()
            R_edge = self.faces['R'][:, 0].copy()
            D_edge = self.faces['D'][0, :].copy()
            L_edge = self.faces['L'][:, 1].copy()

            if direction == 'clockwise':
                self.faces['U'][1, :] = L_edge[::-1]
                self.faces['L'][:, 1] = D_edge
                self.faces['D'][0, :] = R_edge[::-1]
                self.faces['R'][:, 0] = U_edge
            else:
                self.faces['U'][1, :] = R_edge
                self.faces['R'][:, 0] = D_edge[::-1]
                self.faces['D'][0, :] = L_edge
                self.faces['L'][:, 1] = U_edge[::-1]

        elif face == 'B':
            U_edge = self.faces['U'][0, :].copy()
            L_edge = self.faces['L'][:, 0].copy()
            D_edge = self.faces['D'][1, :].copy()
            R_edge = self.faces['R'][:, 1].copy()

            if direction == 'clockwise':
                self.faces['U'][0, :] = R_edge
                self.faces['R'][:, 1] = D_edge[::-1]
                self.faces['D'][1, :] = L_edge
                self.faces['L'][:, 0] = U_edge[::-1]
            else:
                self.faces['U'][0, :] = L_edge[::-1]
                self.faces['L'][:, 0] = D_edge
                self.faces['D'][1, :] = R_edge[::-1]
                self.faces['R'][:, 1] = U_edge

        elif face == 'U':
            B_edge = self.faces['B'][0, :].copy()
            R_edge = self.faces['R'][0, :].copy()
            F_edge = self.faces['F'][0, :].copy()
            L_edge = self.faces['L'][0, :].copy()

            if direction == 'clockwise':
                self.faces['F'][0, :] = R_edge
                self.faces['R'][0, :] = B_edge
                self.faces['B'][0, :] = L_edge
                self.faces['L'][0, :] = F_edge
            else:
                self.faces['F'][0, :] = L_edge
                self.faces['L'][0, :] = B_edge
                self.faces['B'][0, :] = R_edge
                self.faces['R'][0, :] = F_edge

        elif face == 'D':
            F_edge = self.faces['F'][1, :].copy()
            R_edge = self.faces['R'][1, :].copy()
            B_edge = self.faces['B'][1, :].copy()
            L_edge = self.faces['L'][1, :].copy()

            if direction == 'clockwise':
                self.faces['F'][1, :] = L_edge
                self.faces['L'][1, :] = B_edge
                self.faces['B'][1, :] = R_edge
                self.faces['R'][1, :] = F_edge
            else:
                self.faces['F'][1, :] = R_edge
                self.faces['R'][1, :] = B_edge
                self.faces['B'][1, :] = L_edge
                self.faces['L'][1, :] = F_edge

        elif face == 'L':
            U_edge = self.faces['U'][:, 0].copy()
            F_edge = self.faces['F'][:, 0].copy()
            D_edge = self.faces['D'][:, 0].copy()
            B_edge = self.faces['B'][:, 1].copy()

            if direction == 'clockwise':
                self.faces['F'][:, 0] = D_edge
                self.faces['D'][:, 0] = B_edge[::-1]
                self.faces['B'][:, 1] = U_edge[::-1]
                self.faces['U'][:, 0] = F_edge
            else:
                self.faces['F'][:, 0] = U_edge
                self.faces['U'][:, 0] = B_edge[::-1]
                self.faces['B'][:, 1] = D_edge[::-1]
                self.faces['D'][:, 0] = F_edge

        elif face == 'R':
            U_edge = self.faces['U'][:, 1].copy()
            B_edge = self.faces['B'][:, 0].copy()
            D_edge = self.faces['D'][:, 1].copy()
            F_edge = self.faces['F'][:, 1].copy()

            if direction == 'clockwise':
                self.faces['F'][:, 1] = U_edge
                self.faces['U'][:, 1] = B_edge[::-1]
                self.faces['B'][:, 0] = D_edge[::-1]
                self.faces['D'][:, 1] = F_edge
            else:
                self.faces['F'][:, 1] = D_edge
                self.faces['D'][:, 1] = B_edge[::-1]
                self.faces['B'][:, 0] = U_edge[::-1]
                self.faces['U'][:, 1] = F_edge

    def scramble(self, n=8):
        """Embaralha o cubo com n movimentos aleatórios (reduzido para 2x2)"""
        self.move_history = []
        moves = self.get_possible_moves()
        for _ in range(n):
            move = random.choice(moves)
            self._apply_move_instantly(move)
            direction_symbol = "" if move[1] == 'clockwise' else "'"
            self.move_history.append(f"{move[0]}{direction_symbol}")

    def _apply_move_instantly(self, move):
        """Aplica um movimento instantaneamente (sem animação)"""
        face, direction = move
        face_data = self.faces[face]
        if direction == 'clockwise':
            self.faces[face] = np.rot90(face_data, 3)
        else:
            self.faces[face] = np.rot90(face_data, 1)
        self._rotate_adjacent_edges_2x2(face, direction)

    def get_possible_moves(self):
        """Retorna todos os movimentos possíveis"""
        faces = ['F', 'B', 'U', 'D', 'L', 'R']
        directions = ['clockwise', 'counterclockwise']
        return [(face, direction) for face in faces for direction in directions]

    def count_solved_pieces(self):
        """Conta quantas peças estão na posição correta"""
        count = 0
        for face_name, face in self.faces.items():
            target_color = face[0, 0]
            for i in range(self.size):
                for j in range(self.size):
                    if face[i, j] == target_color:
                        count += 1
        return count

    def print_moves(self):
        """Imprime os movimentos realizados no console"""
        if not self.move_history:
            print("Nenhum movimento realizado ainda.")
            return
        print("\nHistórico de Movimentos:")
        print(" ".join(self.move_history))
        print(f"Total de movimentos: {len(self.move_history)}")


# 2. Ambiente RL Melhorado
class RubiksEnv:
    def __init__(self):
        self.cube = RubiksCube3D(size=2)
        self.action_space = self.cube.get_possible_moves()
        self.max_moves = 200
        self.moves_count = 0
        self.reset()

    def reset(self):
        self.cube = RubiksCube3D(size=2)
        self.cube.scramble(6)
        self.moves_count = 0
        return self.cube.get_state()

    def step(self, action):
        """Executa uma ação e retorna (next_state, reward, done)"""
        old_solved_pieces = self.cube.count_solved_pieces()
        self.cube._apply_move_instantly(action)
        self.moves_count += 1
        new_state = self.cube.get_state()
        new_solved_pieces = self.cube.count_solved_pieces()
        reward = self._calculate_reward(old_solved_pieces, new_solved_pieces)
        done = self.cube.is_solved() or self.moves_count >= self.max_moves
        return new_state, reward, done

    def _calculate_reward(self, old_solved, new_solved):
        """Calcula a recompensa baseada na melhoria do estado"""
        if self.cube.is_solved():
            return 1000
        improvement = new_solved - old_solved
        base_reward = improvement * 10 - 1
        if new_solved >= 20:
            base_reward += 50
        elif new_solved >= 16:
            base_reward += 20
        return base_reward


# 3. Algoritmo Q-Learning Melhorado com Parâmetros Ajustáveis
class QLearningAgent:
    def __init__(self, env, max_episodes=1000, learning_rate=0.2, discount_factor=0.9,
                 exploration_rate=0.8, exploration_decay=0.998, min_exploration=0.05):
        self.env = env
        self.q_table = defaultdict(lambda: np.zeros(len(env.action_space)))
        self.max_episodes = max_episodes
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration = min_exploration
        self.episodes_completed = 0
        self.best_score = 0
        self.solved_count = 0
        self.training_stopped = False

    def get_action(self, state):
        """Escolhe uma ação usando ε-greedy melhorado"""
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
        self.exploration_rate = max(self.min_exploration,
                                    self.exploration_rate * self.exploration_decay)

    def episode_finished(self, solved, moves, final_score):
        """Chamado quando um episódio termina"""
        self.episodes_completed += 1
        if solved:
            self.solved_count += 1
        if final_score > self.best_score:
            self.best_score = final_score
        if self.episodes_completed >= self.max_episodes:
            self.training_stopped = True
            print(f"🤖 Treinamento concluído após {self.max_episodes} episódios")

    def stop_training(self):
        """Para o treinamento"""
        self.training_stopped = True


# 4. Interface Gráfica 3D com Controles de Teclado e Mouse
class Rubiks3DGUI:
    def __init__(self, agent):
        pygame.init()
        self.width, self.height = 800, 600
        self.screen = pygame.display.set_mode((self.width, self.height), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Q-Learning Cubo Mágico 2x2x2 (Controles: Teclado e Mouse)")

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
            'W': (1, 1, 1),  # Branco
            'Y': (1, 1, 0),  # Amarelo
            'G': (0, 0.8, 0),  # Verde
            'B': (0, 0, 1),  # Azul
            'O': (1, 0.5, 0),  # Laranja
            'R': (1, 0, 0)  # Vermelho
        }

        self.rotation_x = 20
        self.rotation_y = -30
        self.last_pos = None

        self.training = False
        self.training_speed = 0.05
        self.running = True

    def draw_cube(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, -6)
        glRotatef(self.rotation_x, 1, 0, 0)
        glRotatef(self.rotation_y, 0, 1, 0)

        cube_size = 1.5
        piece_size = cube_size / 2
        offset = piece_size / 2

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
        colors = self.colors

        front_color = self.cube.faces['F'][y, x] if z == 1 else None
        back_color = self.cube.faces['B'][1 - y, x] if z == 0 else None
        up_color = self.cube.faces['U'][1 - x, z] if y == 1 else None
        down_color = self.cube.faces['D'][x, z] if y == 0 else None
        left_color = self.cube.faces['L'][1 - y, z] if x == 0 else None
        right_color = self.cube.faces['R'][y, z] if x == 1 else None

        glBegin(GL_QUADS)
        if front_color:
            glColor3fv(colors[front_color])
            glVertex3f(-0.5 * piece_size, -0.5 * piece_size, 0.5 * piece_size)
            glVertex3f(0.5 * piece_size, -0.5 * piece_size, 0.5 * piece_size)
            glVertex3f(0.5 * piece_size, 0.5 * piece_size, 0.5 * piece_size)
            glVertex3f(-0.5 * piece_size, 0.5 * piece_size, 0.5 * piece_size)
        if back_color:
            glColor3fv(colors[back_color])
            glVertex3f(-0.5 * piece_size, -0.5 * piece_size, -0.5 * piece_size)
            glVertex3f(-0.5 * piece_size, 0.5 * piece_size, -0.5 * piece_size)
            glVertex3f(0.5 * piece_size, 0.5 * piece_size, -0.5 * piece_size)
            glVertex3f(0.5 * piece_size, -0.5 * piece_size, -0.5 * piece_size)
        if up_color:
            glColor3fv(colors[up_color])
            glVertex3f(-0.5 * piece_size, 0.5 * piece_size, -0.5 * piece_size)
            glVertex3f(-0.5 * piece_size, 0.5 * piece_size, 0.5 * piece_size)
            glVertex3f(0.5 * piece_size, 0.5 * piece_size, 0.5 * piece_size)
            glVertex3f(0.5 * piece_size, 0.5 * piece_size, -0.5 * piece_size)
        if down_color:
            glColor3fv(colors[down_color])
            glVertex3f(-0.5 * piece_size, -0.5 * piece_size, -0.5 * piece_size)
            glVertex3f(0.5 * piece_size, -0.5 * piece_size, -0.5 * piece_size)
            glVertex3f(0.5 * piece_size, -0.5 * piece_size, 0.5 * piece_size)
            glVertex3f(-0.5 * piece_size, -0.5 * piece_size, 0.5 * piece_size)
        if left_color:
            glColor3fv(colors[left_color])
            glVertex3f(-0.5 * piece_size, -0.5 * piece_size, -0.5 * piece_size)
            glVertex3f(-0.5 * piece_size, -0.5 * piece_size, 0.5 * piece_size)
            glVertex3f(-0.5 * piece_size, 0.5 * piece_size, 0.5 * piece_size)
            glVertex3f(-0.5 * piece_size, 0.5 * piece_size, -0.5 * piece_size)
        if right_color:
            glColor3fv(colors[right_color])
            glVertex3f(0.5 * piece_size, -0.5 * piece_size, -0.5 * piece_size)
            glVertex3f(0.5 * piece_size, 0.5 * piece_size, -0.5 * piece_size)
            glVertex3f(0.5 * piece_size, 0.5 * piece_size, 0.5 * piece_size)
            glVertex3f(0.5 * piece_size, -0.5 * piece_size, 0.5 * piece_size)
        glEnd()

        glColor3f(0, 0, 0)
        glBegin(GL_LINES)
        for edge in [
            ((-0.5 * piece_size, -0.5 * piece_size, 0.5 * piece_size),
             (0.5 * piece_size, -0.5 * piece_size, 0.5 * piece_size)),
            ((0.5 * piece_size, -0.5 * piece_size, 0.5 * piece_size),
             (0.5 * piece_size, 0.5 * piece_size, 0.5 * piece_size)),
            ((0.5 * piece_size, 0.5 * piece_size, 0.5 * piece_size),
             (-0.5 * piece_size, 0.5 * piece_size, 0.5 * piece_size)),
            ((-0.5 * piece_size, 0.5 * piece_size, 0.5 * piece_size),
             (-0.5 * piece_size, -0.5 * piece_size, 0.5 * piece_size)),
            ((-0.5 * piece_size, -0.5 * piece_size, -0.5 * piece_size),
             (0.5 * piece_size, -0.5 * piece_size, -0.5 * piece_size)),
            ((0.5 * piece_size, -0.5 * piece_size, -0.5 * piece_size),
             (0.5 * piece_size, 0.5 * piece_size, -0.5 * piece_size)),
            ((0.5 * piece_size, 0.5 * piece_size, -0.5 * piece_size),
             (-0.5 * piece_size, 0.5 * piece_size, -0.5 * piece_size)),
            ((-0.5 * piece_size, 0.5 * piece_size, -0.5 * piece_size),
             (-0.5 * piece_size, -0.5 * piece_size, -0.5 * piece_size)),
            ((-0.5 * piece_size, -0.5 * piece_size, -0.5 * piece_size),
             (-0.5 * piece_size, -0.5 * piece_size, 0.5 * piece_size)),
            ((0.5 * piece_size, -0.5 * piece_size, -0.5 * piece_size),
             (0.5 * piece_size, -0.5 * piece_size, 0.5 * piece_size)),
            ((0.5 * piece_size, 0.5 * piece_size, -0.5 * piece_size),
             (0.5 * piece_size, 0.5 * piece_size, 0.5 * piece_size)),
            ((-0.5 * piece_size, 0.5 * piece_size, -0.5 * piece_size),
             (-0.5 * piece_size, 0.5 * piece_size, 0.5 * piece_size)),
        ]:
            glVertex3f(*edge[0])
            glVertex3f(*edge[1])
        glEnd()

    def test_solution(self):
        """Testa a solução aprendida pelo agente"""
        print("🧪 Testando solução aprendida")
        self.training = False
        self.agent.training_stopped = True
        self.cube.move_history = []
        state = self.agent.env.reset()
        moves = 0
        max_test_moves = 100

        while not self.cube.is_solved() and moves < max_test_moves:
            action_idx = self.agent.get_action(state)
            action = self.agent.env.action_space[action_idx]
            self.cube.apply_move(action)
            while self.cube.is_animating:
                self.cube.update_animation()
                self.draw_cube()
                pygame.display.flip()
                pygame.time.wait(16)
            state, _, done = self.agent.env.step(action)
            moves += 1
            if done:
                break

        print(f"✅ Teste concluído em {moves} movimentos")
        self.cube.print_moves()

    def run(self):
        """Loop principal do programa com controles de teclado e mouse"""
        clock = pygame.time.Clock()
        last_action_time = time.time()
        episode_moves = 0
        episode_score = 0

        while self.running:
            current_time = time.time()
            for event in pygame.event.get():
                if event.type == QUIT:
                    self.running = False
                elif event.type == KEYDOWN:
                    if event.key == K_SPACE:
                        self.training = not self.training
                        self.agent.training_stopped = not self.training
                        status = "iniciado" if self.training else "parado"
                        print(f"🤖 Treinamento {status}")
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

            if self.training and not self.cube.is_animating and not self.agent.training_stopped:
                if current_time - last_action_time >= self.training_speed:
                    state = self.cube.get_state()
                    action_idx = self.agent.get_action(state)
                    action = self.agent.env.action_space[action_idx]
                    self.cube.apply_move(action)
                    next_state, reward, done = self.agent.env.step(action)
                    self.agent.update(state, action_idx, reward, next_state)
                    episode_score += reward
                    episode_moves += 1

                    if done:
                        self.agent.episode_finished(self.cube.is_solved(), episode_moves, episode_score)
                        state = self.agent.env.reset()
                        episode_moves = 0
                        episode_score = 0
                    last_action_time = current_time

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
            max_episodes=500,
            learning_rate=0.1,
            discount_factor=0.95,
            exploration_rate=0.9,
            exploration_decay=0.995,
            min_exploration=0.01
        )
        gui = Rubiks3DGUI(agent)
        gui.run()
    except Exception as e:
        print(f"Erro ao executar o programa: {e}")
        pygame.quit()
        raise