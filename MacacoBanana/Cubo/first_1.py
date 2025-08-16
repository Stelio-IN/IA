
import asyncio
import platform
import pygame
from pygame.locals import *
import numpy as np
import random
from collections import defaultdict
from OpenGL.GL import *
from OpenGL.GLU import *

# Define positions and distances
positions = ["canto1", "canto2", "canto3", "canto4", "centro"]
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
ACTION_COSTS = {
    "mover": 1.0,
    "pegar_vara": 2.0,
    "colocar_vara": 1.5,
    "empurrar_cadeira": 4.0,
    "subir_cadeira": 3.0,
    "apontar_vara": 2.5,
    "agitar_vara": 2.0
}

# Position coordinates for visualization (scaled to 800x600 window)
POS_COORDS = {
    "canto1": (100, 500),  # Bottom-left
    "canto2": (100, 100),  # Top-left
    "canto3": (700, 100),  # Top-right
    "canto4": (700, 500),  # Bottom-right
    "centro": (400, 300)   # Center
}

FPS = 60
FRAME_DELAY = 0.02  # 20ms per frame
ACTION_DURATION = 1.0  # 1 second per action

def dist(a, b):
    if a == b:
        return 0
    return distances.get((a, b), 100)

def print_action(step, action_name, reward, estado):
    print(f"{step}. {action_name:25} | Recompensa: {reward:7.2f} | Estado: {estado}")

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
        return (
            self.pos_macaco, self.pos_cadeira, self.pos_vara, self.pos_bananas,
            self.tem_vara, self.em_cima, self.tem_bananas, self.vara_na_cadeira, self.vara_apontada
        )

def objetivo(estado):
    return estado.tem_bananas

def obter_acoes_possiveis(estado):
    acoes_possiveis = []
    for p in positions:
        if p != estado.pos_macaco:
            novo = Estado(p, estado.pos_cadeira, estado.pos_vara, estado.pos_bananas, estado.tem_vara, False,
                          estado.tem_bananas, estado.vara_na_cadeira, estado.vara_apontada)
            recompensa = -dist(estado.pos_macaco, p) * ACTION_COSTS["mover"]
            acoes_possiveis.append(("ir_para_" + p, novo, recompensa))
    if estado.pos_macaco == estado.pos_vara and not estado.tem_vara and not estado.vara_na_cadeira and not estado.em_cima:
        novo = Estado(estado.pos_macaco, estado.pos_cadeira, estado.pos_macaco, estado.pos_bananas, True,
                      estado.em_cima, estado.tem_bananas, False, estado.vara_apontada)
        acoes_possiveis.append(("pegar_vara", novo, -ACTION_COSTS["pegar_vara"]))
    if estado.tem_vara and estado.pos_macaco == estado.pos_cadeira and not estado.em_cima:
        novo = Estado(estado.pos_macaco, estado.pos_cadeira, estado.pos_cadeira, estado.pos_bananas, False,
                      estado.em_cima, estado.tem_bananas, True, estado.vara_apontada)
        acoes_possiveis.append(("colocar_vara_sobre_cadeira", novo, -ACTION_COSTS["colocar_vara"]))
    if estado.vara_na_cadeira and estado.pos_macaco == estado.pos_cadeira and estado.em_cima:
        novo = Estado(estado.pos_macaco, estado.pos_cadeira, estado.pos_cadeira, estado.pos_bananas, True, True,
                      estado.tem_bananas, False, estado.vara_apontada)
        acoes_possiveis.append(("pegar_vara_em_cima_cadeira", novo, -ACTION_COSTS["pegar_vara"]))
    if estado.tem_vara and estado.em_cima and estado.pos_macaco == estado.pos_bananas:
        novo = Estado(estado.pos_macaco, estado.pos_cadeira, estado.pos_vara, estado.pos_bananas, True, True,
                      estado.tem_bananas, estado.vara_na_cadeira, True)
        acoes_possiveis.append(("apontar_vara", novo, -ACTION_COSTS["apontar_vara"]))
    if estado.vara_apontada and estado.tem_vara and estado.em_cima and estado.pos_macaco == estado.pos_bananas:
        novo = Estado(estado.pos_macaco, estado.pos_cadeira, estado.pos_vara, estado.pos_bananas, True, True, True,
                      estado.vara_na_cadeira, True)
        acoes_possiveis.append(("agitar_vara", novo, 100 - ACTION_COSTS["agitar_vara"]))
    if estado.pos_macaco == estado.pos_cadeira and not estado.em_cima:
        for p in positions:
            if p != estado.pos_macaco:
                novo = Estado(p, p, estado.pos_vara, estado.pos_bananas, estado.tem_vara, False, estado.tem_bananas,
                              estado.vara_na_cadeira, estado.vara_apontada)
                recompensa = -(dist(estado.pos_macaco, p) * ACTION_COSTS["mover"] + ACTION_COSTS["empurrar_cadeira"])
                acoes_possiveis.append(("empurrar_cadeira_para_" + p, novo, recompensa))
    if estado.pos_macaco == estado.pos_cadeira and not estado.em_cima:
        novo = Estado(estado.pos_macaco, estado.pos_cadeira, estado.pos_vara, estado.pos_bananas, estado.tem_vara, True,
                      estado.tem_bananas, estado.vara_na_cadeira, estado.vara_apontada)
        acoes_possiveis.append(("subir_cadeira", novo, -ACTION_COSTS["subir_cadeira"]))
    return acoes_possiveis

class MonkeyQLearning:
    def __init__(self, estado_inicial, alpha=0.1, gamma=0.9, epsilon=0.3, epsilon_decay=0.99, min_epsilon=0.01):
        self.estado_inicial = estado_inicial
        self.q_values = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.rewards_history = []

    def get_q_value(self, estado_tuple, acao):
        if (estado_tuple, acao) not in self.q_values:
            self.q_values[(estado_tuple, acao)] = 0.0
        return self.q_values[(estado_tuple, acao)]

    def choose_action(self, estado):
        acoes_possiveis = obter_acoes_possiveis(estado)
        if not acoes_possiveis:
            return None, None, None
        if random.random() < self.epsilon:
            return random.choice(acoes_possiveis)
        estado_tuple = estado.to_tuple()
        q_values = [(self.get_q_value(estado_tuple, acao[0]), acao) for acao in acoes_possiveis]
        max_q_value = max(q_values, key=lambda x: x[0])[0]
        best_actions = [acao for q, acao in q_values if q == max_q_value]
        return random.choice(best_actions)

    def update_q_value(self, estado_tuple, acao, recompensa, proximo_estado_tuple, proximas_acoes):
        max_q_next = max([self.get_q_value(proximo_estado_tuple, a[0]) for a in proximas_acoes], default=0) if proximas_acoes else 0
        current_q = self.get_q_value(estado_tuple, acao)
        new_q = current_q + self.alpha * (recompensa + self.gamma * max_q_next - current_q)
        self.q_values[(estado_tuple, acao)] = new_q

    async def treinar(self, num_episodios=1000, max_steps=100, verbose=False):
        episodios_completos = []
        recompensas_por_episodio = []
        passos_por_episodio = []

        print("\n=== INICIANDO TREINAMENTO ===")
        print(f"Parâmetros: α={self.alpha:.2f}, γ={self.gamma:.2f}, ε={self.epsilon:.2f}")
        print(f"Estado inicial: {self.estado_inicial}\n")

        for episodio in range(1, num_episodios + 1):
            estado = Estado(
                self.estado_inicial.pos_macaco,
                self.estado_inicial.pos_cadeira,
                self.estado_inicial.pos_vara,
                self.estado_inicial.pos_bananas,
                self.estado_inicial.tem_vara,
                self.estado_inicial.em_cima,
                self.estado_inicial.tem_bananas,
                self.estado_inicial.vara_na_cadeira,
                self.estado_inicial.vara_apontada
            )
            recompensa_total = 0
            step = 0

            if verbose or episodio % 100 == 0:
                print(f"\nEpisódio {episodio}/{num_episodios}, ε={self.epsilon:.3f}")

            while step < max_steps and not objetivo(estado):
                resultado_acao = self.choose_action(estado)
                if not resultado_acao:
                    if verbose:
                        print("  Sem ações possíveis!")
                    break
                nome_acao, proximo_estado, recompensa = resultado_acao
                if verbose or episodio % 100 == 0:
                    print_action(step + 1, nome_acao, recompensa, proximo_estado)
                estado_tuple = estado.to_tuple()
                proximo_estado_tuple = proximo_estado.to_tuple()
                proximas_acoes = obter_acoes_possiveis(proximo_estado)
                self.update_q_value(estado_tuple, nome_acao, recompensa, proximo_estado_tuple, proximas_acoes)
                estado = proximo_estado
                recompensa_total += recompensa
                step += 1
            recompensas_por_episodio.append(recompensa_total)
            passos_por_episodio.append(step)
            if objetivo(estado):
                episodios_completos.append(episodio)
                if verbose or episodio % 100 == 0:
                    print(f"✅ Objetivo alcançado em {step} passos! Recompensa total: {recompensa_total:.2f}")
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        taxa_sucesso = len(episodios_completos) / num_episodios * 100
        print("\n=== TREINAMENTO CONCLUÍDO ===")
        print(f"Episódios com sucesso: {len(episodios_completos)}/{num_episodios} ({taxa_sucesso:.1f}%)")
        if episodios_completos:
            ultimos_100 = passos_por_episodio[-100:] if len(passos_por_episodio) >= 100 else passos_por_episodio
            media_passos = sum(ultimos_100) / len(ultimos_100)
            print(f"Média de passos (últimos {len(ultimos_100)} episódios): {media_passos:.1f}")
        self.rewards_history = recompensas_por_episodio
        return episodios_completos, recompensas_por_episodio, passos_por_episodio

    async def executar_politica(self, estado_inicial, max_steps=100, visualizer=None):
        estado = estado_inicial
        plano = []
        custo_total = 0

        print("\n=== EXECUTANDO POLÍTICA APRENDIDA ===")
        print(f"Estado inicial: {estado}\n")

        for step in range(max_steps):
            if objetivo(estado):
                print(f"✅ Objetivo alcançado em {step} passos!")
                print(f"Custo total da solução: {custo_total:.2f}")
                if visualizer:
                    visualizer.update_display(estado, f"Objetivo alcançado! Custo: {custo_total:.2f}", step + 1, custo_total)
                    await asyncio.sleep(ACTION_DURATION)
                break
            acoes_possiveis = obter_acoes_possiveis(estado)
            if not acoes_possiveis:
                print("❌ Sem ações possíveis!")
                if visualizer:
                    visualizer.update_display(estado, "Sem ações possíveis!", step + 1, custo_total)
                    await asyncio.sleep(ACTION_DURATION)
                break
            estado_tuple = estado.to_tuple()
            q_values = [(self.get_q_value(estado_tuple, acao[0]), acao) for acao in acoes_possiveis]
            _, (nome_acao, proximo_estado, recompensa) = max(q_values, key=lambda x: x[0])
            custo_acao = -recompensa if "agitar_vara" not in nome_acao else ACTION_COSTS["agitar_vara"]
            custo_total += custo_acao
            plano.append((nome_acao, proximo_estado, custo_acao))
            print_action(step + 1, nome_acao, recompensa, proximo_estado)
            print(f"   Custo acumulado: {custo_total:.2f}")
            if visualizer:
                await visualizer.animate_action(estado, proximo_estado, nome_acao, step + 1, custo_total)
            estado = proximo_estado
        if not objetivo(estado):
            print(f"❌ Política não conseguiu atingir o objetivo em {max_steps} passos")
            return None
        return plano

class Visualizer:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600), pygame.OPENGL | pygame.DOUBLEBUF)
        pygame.display.set_caption("Monkey and Banana Problem")
        glClearColor(0.9, 0.9, 0.9, 1.0)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        self.font = pygame.font.SysFont('Arial', 24)
        self.monkey_pos = [400, 300]
        self.chair_pos = [400, 300]
        self.stick_pos = [400, 300]
        self.banana_pos = [400, 300]
        self.clock = pygame.time.Clock()

    def draw_square(self, x, y, size, color):
        glBegin(GL_QUADS)
        glColor4f(*color)
        glVertex2f(x - size, y - size)
        glVertex2f(x + size, y - size)
        glVertex2f(x + size, y + size)
        glVertex2f(x - size, y + size)
        glEnd()

    def draw_rectangle(self, x, y, width, height, color):
        glBegin(GL_QUADS)
        glColor4f(*color)
        glVertex2f(x - width / 2, y - height / 2)
        glVertex2f(x + width / 2, y - height / 2)
        glVertex2f(x + width / 2, y + height / 2)
        glVertex2f(x - width / 2, y + height / 2)
        glEnd()

    def draw_text(self, text, x, y, color=(0, 0, 0, 1)):
        surface = self.font.render(text, True, color)
        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, surface.get_width(), surface.get_height(), 0, GL_RGBA, GL_UNSIGNED_BYTE, pygame.image.tostring(surface, 'RGBA'))
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glEnable(GL_TEXTURE_2D)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 1); glVertex2f(x, y)
        glTexCoord2f(1, 1); glVertex2f(x + surface.get_width(), y)
        glTexCoord2f(1, 0); glVertex2f(x + surface.get_width(), y + surface.get_height())
        glTexCoord2f(0, 0); glVertex2f(x, y + surface.get_height())
        glEnd()
        glDeleteTextures(1, [texture])
        glDisable(GL_TEXTURE_2D)

    def setup(self):
        glMatrixMode(GL_PROJECTION)
        gluOrtho2D(0, 800, 0, 600)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    async def animate_action(self, estado_atual, proximo_estado, nome_acao, step, custo_total):
        frames = int(ACTION_DURATION / FRAME_DELAY)
        start_pos = np.array(self.monkey_pos)
        end_pos = np.array(POS_COORDS[proximo_estado.pos_macaco])
        chair_start = np.array(self.chair_pos)
        chair_end = np.array(POS_COORDS[proximo_estado.pos_cadeira])
        stick_start = np.array(self.stick_pos)
        stick_end = np.array(POS_COORDS[proximo_estado.pos_vara])
        banana_end = np.array(POS_COORDS[proximo_estado.pos_bananas])
        for i in range(frames):
            t = i / frames
            if nome_acao.startswith("ir_para_") or nome_acao.startswith("empurrar_cadeira_para_"):
                self.monkey_pos = start_pos + t * (end_pos - start_pos)
                if nome_acao.startswith("empurrar_cadeira_para_"):
                    self.chair_pos = chair_start + t * (chair_end - chair_start)
            if nome_acao in ["pegar_vara", "pegar_vara_em_cima_cadeira", "colocar_vara_sobre_cadeira"]:
                self.stick_pos = stick_start + t * (stick_end - stick_start)
            if nome_acao == "agitar_vara":
                # Animate bananas moving to monkey
                self.banana_pos = banana_end if i == frames - 1 else banana_end + t * (banana_end - self.banana_pos)
            self.update_display(proximo_estado, f"Ação: {nome_acao} | Custo: {custo_total:.2f}", step, custo_total)
            await asyncio.sleep(FRAME_DELAY)
        self.monkey_pos = end_pos
        self.chair_pos = chair_end
        self.stick_pos = stick_end
        self.banana_pos = banana_end if nome_acao != "agitar_vara" else end_pos  # Bananas follow monkey after waving

    def update_display(self, estado, action_text, step, custo_total):
        glClear(GL_COLOR_BUFFER_BIT)
        # Draw room corners and center
        for pos, (x, y) in POS_COORDS.items():
            self.draw_square(x, y, 10, (0.5, 0.5, 0.5, 1.0))
        # Draw bananas (unless obtained)
        if not estado.tem_bananas:
            self.draw_square(*self.banana_pos, 20, (1, 1, 0, 1))
        # Draw chair
        self.draw_square(*self.chair_pos, 30, (0.5, 0.25, 0, 1))
        # Draw stick
        if estado.tem_vara:
            y_offset = 40 if estado.em_cima else 20
            angle = 45 if estado.vara_apontada else 0
            glPushMatrix()
            glTranslatef(self.monkey_pos[0], self.monkey_pos[1] + y_offset, 0)
            glRotatef(angle, 0, 0,  ATS, 1)
            self.draw_rectangle(0, 0, 40, 5, (0.6, 0.3, 0, 1))
            glPopMatrix()
        elif estado.vara_na_cadeira:
            self.draw_square(*self.chair_pos, 30, (0.5, 0.25, 0, 1))
            y_offset = 10
            self.draw_rectangle(0, y_offset, 40, 5, (0.6, 0.3, 0, 1))
        else:
            self.draw_rectangle(*self.stick_pos, 40, 5, (0.6, 0.3, 0, 1))
        # Draw monkey
        self.draw_square(*self.monkey_pos, 20, (0, 0, 1, 1) if not estado.em_cima else (0, 0, 0.7, 1))
        # Draw text
        self.draw_text(f"Passo: {step}", 10, 580)
        self.draw_text(action_text, 10, 550)
        self.draw_text(f"Estado: {estado}", 10, 520)
        pygame.display.flip()
        self.clock.tick(FPS)

async def main():
    estado_inicial = Estado("canto1", "canto3", "canto3", "canto3", False, False, False)
    visualizer = Visualizer()
    visualizer.setup()
    agente = MonkeyQLearning(estado_inicial)
    await agente.treinar(num_episodios=1000, verbose=True)
    await agente.executar_politica(estado_inicial, visualizer=visualizer)
    pygame.quit()

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())
