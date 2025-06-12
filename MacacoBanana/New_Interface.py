import tkinter as tk
from tkinter import ttk, messagebox
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
from collections import defaultdict
import matplotlib

matplotlib.use('TkAgg')

# Configura estilo visual dos plots
plt.style.use('ggplot')

# Posições e distâncias (mesmos dados do original)
positions = ["canto1", "canto2", "canto3", "canto4", "centro"]

# Distâncias entre posições
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

# Custos para ações específicas
ACTION_COSTS = {
    "mover": 1.0,
    "pegar_vara": 2.0,
    "colocar_vara": 1.5,
    "empurrar_cadeira": 4.0,
    "subir_cadeira": 3.0,
    "apontar_vara": 2.5,
    "agitar_vara": 2.0
}


def dist(a, b):
    """Retorna a distância entre duas posições."""
    if a == b:
        return 0
    return distances.get((a, b), 100)


def print_action(step, action_name, reward, estado):
    """Helper function to print action details in a consistent format"""
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
        """Converte o estado para uma tupla para uso como chave no dicionário Q."""
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
    """Verifica se o objetivo foi alcançado (macaco tem as bananas)."""
    return estado.tem_bananas


def obter_acoes_possiveis(estado):
    """Retorna lista de tuplas com (nome_acao, novo_estado, recompensa)."""
    acoes_possiveis = []

    # Mover-se
    for p in positions:
        if p != estado.pos_macaco:
            novo = Estado(p, estado.pos_cadeira, estado.pos_vara, estado.pos_bananas, estado.tem_vara, False,
                          estado.tem_bananas, estado.vara_na_cadeira, estado.vara_apontada)
            recompensa = -dist(estado.pos_macaco, p) * ACTION_COSTS["mover"]
            acoes_possiveis.append(("ir_para_" + p, novo, recompensa))

    # Pegar vara do chão
    if estado.pos_macaco == estado.pos_vara and not estado.tem_vara and not estado.vara_na_cadeira and not estado.em_cima:
        novo = Estado(estado.pos_macaco, estado.pos_cadeira, estado.pos_macaco, estado.pos_bananas, True,
                      estado.em_cima, estado.tem_bananas, False, estado.vara_apontada)
        acoes_possiveis.append(("pegar_vara", novo, -ACTION_COSTS["pegar_vara"]))

    # Colocar vara sobre a cadeira
    if estado.tem_vara and estado.pos_macaco == estado.pos_cadeira and not estado.em_cima:
        novo = Estado(estado.pos_macaco, estado.pos_cadeira, estado.pos_cadeira, estado.pos_bananas, False,
                      estado.em_cima, estado.tem_bananas, True, estado.vara_apontada)
        acoes_possiveis.append(("colocar_vara_sobre_cadeira", novo, -ACTION_COSTS["colocar_vara"]))

    # Pegar vara em cima da cadeira
    if estado.vara_na_cadeira and estado.pos_macaco == estado.pos_cadeira and estado.em_cima:
        novo = Estado(estado.pos_macaco, estado.pos_cadeira, estado.pos_cadeira, estado.pos_bananas, True, True,
                      estado.tem_bananas, False, estado.vara_apontada)
        acoes_possiveis.append(("pegar_vara_em_cima_cadeira", novo, -ACTION_COSTS["pegar_vara"]))

    # Apontar vara para as bananas
    if estado.tem_vara and estado.em_cima and estado.pos_macaco == estado.pos_bananas:
        novo = Estado(estado.pos_macaco, estado.pos_cadeira, estado.pos_vara, estado.pos_bananas, True, True,
                      estado.tem_bananas, estado.vara_na_cadeira, True)
        acoes_possiveis.append(("apontar_vara", novo, -ACTION_COSTS["apontar_vara"]))

    # Agitar vara (pegar bananas)
    if estado.vara_apontada and estado.tem_vara and estado.em_cima and estado.pos_macaco == estado.pos_bananas:
        novo = Estado(estado.pos_macaco, estado.pos_cadeira, estado.pos_vara, estado.pos_bananas, True, True, True,
                      estado.vara_na_cadeira, True)
        acoes_possiveis.append(("agitar_vara", novo, 100 - ACTION_COSTS["agitar_vara"]))

    # Empurrar cadeira
    if estado.pos_macaco == estado.pos_cadeira and not estado.em_cima:
        for p in positions:
            if p != estado.pos_macaco:
                novo = Estado(p, p, estado.pos_vara, estado.pos_bananas, estado.tem_vara, False, estado.tem_bananas,
                              estado.vara_na_cadeira, estado.vara_apontada)
                recompensa = -(dist(estado.pos_macaco, p) * ACTION_COSTS["mover"] + ACTION_COSTS["empurrar_cadeira"])
                acoes_possiveis.append(("empurrar_cadeira_para_" + p, novo, recompensa))

    # Subir na cadeira
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
        if not proximas_acoes:
            max_q_next = 0
        else:
            max_q_next = max([self.get_q_value(proximo_estado_tuple, a[0]) for a in proximas_acoes], default=0)

        current_q = self.get_q_value(estado_tuple, acao)
        new_q = current_q + self.alpha * (recompensa + self.gamma * max_q_next - current_q)
        self.q_values[(estado_tuple, acao)] = new_q

    def treinar(self, num_episodios=1000, max_steps=100, verbose=False, progress_callback=None):
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

            # Update progress bar
            if progress_callback and episodio % 10 == 0:
                progress = (episodio / num_episodios) * 100
                progress_callback(progress)

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
                self.update_q_value(estado_tuple, nome_acao, recompensa,
                                    proximo_estado_tuple, proximas_acoes)

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

        # Ensure progress bar is completed
        if progress_callback:
            progress_callback(100)

        taxa_sucesso = len(episodios_completos) / num_episodios * 100
        print("\n=== TREINAMENTO CONCLUÍDO ===")
        print(f"Episódios com sucesso: {len(episodios_completos)}/{num_episodios} ({taxa_sucesso:.1f}%)")

        if episodios_completos:
            ultimos_100 = passos_por_episodio[-100:] if len(passos_por_episodio) >= 100 else passos_por_episodio
            media_passos = sum(ultimos_100) / len(ultimos_100)
            print(f"Média de passos (últimos {len(ultimos_100)} episódios): {media_passos:.1f}")

        self.rewards_history = recompensas_por_episodio
        return episodios_completos, recompensas_por_episodio, passos_por_episodio

    def executar_politica(self, estado_inicial, max_steps=100):
        estado = estado_inicial
        plano = []
        custo_total = 0

        print("\n=== EXECUTANDO POLÍTICA APRENDIDA ===")
        print(f"Estado inicial: {estado}\n")

        for step in range(max_steps):
            if objetivo(estado):
                print(f"✅ Objetivo alcançado em {step} passos!")
                print(f"Custo total da solução: {custo_total:.2f}")
                break

            acoes_possiveis = obter_acoes_possiveis(estado)
            if not acoes_possiveis:
                print("❌ Sem ações possíveis!")
                break

            estado_tuple = estado.to_tuple()
            q_values = [(self.get_q_value(estado_tuple, acao[0]), acao) for acao in acoes_possiveis]
            _, (nome_acao, proximo_estado, recompensa) = max(q_values, key=lambda x: x[0])

            custo_acao = -recompensa if "agitar_vara" not in nome_acao else ACTION_COSTS["agitar_vara"]
            custo_total += custo_acao
            plano.append((nome_acao, proximo_estado, custo_acao))

            print_action(step + 1, nome_acao, recompensa, proximo_estado)
            print(f"   Custo acumulado: {custo_total:.2f}")

            estado = proximo_estado

        if not objetivo(estado):
            print(f"❌ Política não conseguiu atingir o objetivo em {max_steps} passos")
            return None

        return plano


class ModernMonkeyBananaGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Q-Learning: Problema do Macaco e as Bananas")
        self.root.geometry("1200x800")

        # Configure theme and style
        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")

        # Custom styles
        style.configure("TFrame", background="#f5f5f5")
        style.configure("TNotebook", background="#f5f5f5")
        style.configure("TNotebook.Tab", background="#e0e0e0", padding=[10, 5], font=('Helvetica', 10))
        style.map("TNotebook.Tab", background=[("selected", "#4a6ea9"), ("active", "#6989bb")],
                  foreground=[("selected", "#ffffff"), ("active", "#ffffff")])

        style.configure("Header.TLabel", font=('Helvetica', 12, 'bold'), background="#f5f5f5")
        style.configure("Status.TLabel", font=('Helvetica', 10), background="#f5f5f5")
        style.configure("Title.TLabel", font=('Helvetica', 18, 'bold'), background="#f5f5f5", foreground="#333333")

        style.configure("Primary.TButton", font=('Helvetica', 10, 'bold'), background="#4a6ea9", foreground="#ffffff")
        style.map("Primary.TButton", background=[("active", "#6989bb")])

        style.configure("Success.TButton", font=('Helvetica', 10, 'bold'), background="#28a745", foreground="#ffffff")
        style.map("Success.TButton", background=[("active", "#218838")])

        # Variables
        self.pos_macaco = tk.StringVar(value="canto1")
        self.pos_cadeira = tk.StringVar(value="canto3")
        self.pos_vara = tk.StringVar(value="canto3")
        self.pos_bananas = tk.StringVar(value="canto3")

        self.num_episodios = tk.IntVar(value=1000)
        self.alpha = tk.DoubleVar(value=0.1)
        self.gamma = tk.DoubleVar(value=0.9)
        self.epsilon = tk.DoubleVar(value=0.3)

        self.agente = None

        # Main container
        main_frame = ttk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Title
        ttk.Label(main_frame, text="Simulação do Problema do Macaco e as Bananas",
                  style="Title.TLabel").pack(pady=(0, 20))

        # Main notebook for tabs
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True)

        config_frame = ttk.Frame(notebook)
        result_frame = ttk.Frame(notebook)
        graph_frame = ttk.Frame(notebook)

        notebook.add(config_frame, text="Configuração")
        notebook.add(result_frame, text="Resultados")
        notebook.add(graph_frame, text="Gráficos")

        # ================ CONFIGURATION TAB ================
        config_left_frame = ttk.Frame(config_frame)
        config_left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        config_right_frame = ttk.Frame(config_frame)
        config_right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))

        # Initial State Configuration
        estado_frame = ttk.LabelFrame(config_left_frame, text="Cenário Inicial")
        estado_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Grid layout with better spacing
        for i in range(4):
            estado_frame.grid_columnconfigure(i, weight=1, pad=10)

        # Visual diagram of the scenario
        scenario_canvas = tk.Canvas(estado_frame, width=300, height=250, bg="#ffffff", highlightthickness=1,
                                    highlightbackground="#cccccc")
        scenario_canvas.grid(row=0, column=0, columnspan=4, padx=10, pady=10, sticky="nsew")

        # Draw the room layout
        scenario_canvas.create_rectangle(50, 50, 250, 250, outline="#000000")
        scenario_canvas.create_text(40, 40, text="canto1", anchor=tk.NE)
        scenario_canvas.create_text(260, 40, text="canto2", anchor=tk.NW)
        scenario_canvas.create_text(40, 260, text="canto4", anchor=tk.SE)
        scenario_canvas.create_text(260, 260, text="canto3", anchor=tk.SW)
        scenario_canvas.create_text(150, 150, text="centro")

        # Position inputs with better labels and placement
        ttk.Label(estado_frame, text="Posição do Macaco:", style="Header.TLabel").grid(row=1, column=0, sticky=tk.W,
                                                                                       padx=10, pady=(10, 5))
        ttk.Combobox(estado_frame, textvariable=self.pos_macaco, values=positions, state="readonly", width=10).grid(
            row=1, column=1, sticky=tk.W, padx=10, pady=(10, 5))

        ttk.Label(estado_frame, text="Posição da Cadeira:", style="Header.TLabel").grid(row=2, column=0, sticky=tk.W,
                                                                                        padx=10, pady=5)
        ttk.Combobox(estado_frame, textvariable=self.pos_cadeira, values=positions, state="readonly", width=10).grid(
            row=2, column=1, sticky=tk.W, padx=10, pady=5)

        ttk.Label(estado_frame, text="Posição da Vara:", style="Header.TLabel").grid(row=1, column=2, sticky=tk.W,
                                                                                     padx=10, pady=(10, 5))
        ttk.Combobox(estado_frame, textvariable=self.pos_vara, values=positions, state="readonly", width=10).grid(row=1,
                                                                                                                  column=3,
                                                                                                                  sticky=tk.W,
                                                                                                                  padx=10,
                                                                                                                  pady=(
                                                                                                                      10,
                                                                                                                      5))

        ttk.Label(estado_frame, text="Posição das Bananas:", style="Header.TLabel").grid(row=2, column=2, sticky=tk.W,
                                                                                         padx=10, pady=5)
        ttk.Combobox(estado_frame, textvariable=self.pos_bananas, values=positions, state="readonly", width=10).grid(
            row=2, column=3, sticky=tk.W, padx=10, pady=5)

        # Parameters
        params_frame = ttk.LabelFrame(config_right_frame, text="Parâmetros do Q-Learning")
        params_frame.pack(fill=tk.BOTH, expand=True)

        # Add some explanation text
        param_info = tk.Text(params_frame, wrap=tk.WORD, height=5, width=40, font=('Helvetica', 9),
                             bg="#f8f9fa", relief=tk.FLAT)
        param_info.insert(tk.END, "O Q-Learning é um algoritmo de aprendizado por reforço que aprende uma política "
                                  "ótima com base na experiência. Os parâmetros abaixo influenciam diretamente como o "
                                  "agente aprende.")
        param_info.config(state=tk.DISABLED)
        param_info.pack(fill=tk.X, padx=10, pady=10)

        # Parameter sliders instead of entry fields
        ttk.Label(params_frame, text="Número de Episódios:", style="Header.TLabel").pack(anchor=tk.W, padx=10,
                                                                                         pady=(10, 0))
        episodios_scale = ttk.Scale(params_frame, from_=100, to=5000, variable=self.num_episodios, orient=tk.HORIZONTAL)
        episodios_scale.pack(fill=tk.X, padx=10, pady=(0, 10))
        episodios_value = ttk.Label(params_frame, textvariable=self.num_episodios, style="Status.TLabel")
        episodios_value.pack(anchor=tk.E, padx=10)

        ttk.Label(params_frame, text="Taxa de Aprendizado (α):", style="Header.TLabel").pack(anchor=tk.W, padx=10,
                                                                                             pady=(10, 0))
        alpha_frame = ttk.Frame(params_frame)
        alpha_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        alpha_scale = ttk.Scale(alpha_frame, from_=0.01, to=1.0, variable=self.alpha, orient=tk.HORIZONTAL)
        alpha_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(alpha_frame, textvariable=self.alpha, width=5).pack(side=tk.RIGHT)

        ttk.Label(params_frame, text="Fator de Desconto (γ):", style="Header.TLabel").pack(anchor=tk.W, padx=10,
                                                                                           pady=(10, 0))
        gamma_frame = ttk.Frame(params_frame)
        gamma_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        gamma_scale = ttk.Scale(gamma_frame, from_=0.1, to=0.99, variable=self.gamma, orient=tk.HORIZONTAL)
        gamma_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(gamma_frame, textvariable=self.gamma, width=5).pack(side=tk.RIGHT)

        ttk.Label(params_frame, text="Taxa de Exploração (ε):", style="Header.TLabel").pack(anchor=tk.W, padx=10,
                                                                                            pady=(10, 0))
        epsilon_frame = ttk.Frame(params_frame)
        epsilon_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        epsilon_scale = ttk.Scale(epsilon_frame, from_=0.01, to=1.0, variable=self.epsilon, orient=tk.HORIZONTAL)
        epsilon_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(epsilon_frame, textvariable=self.epsilon, width=5).pack(side=tk.RIGHT)

        # Action buttons and status bar at the bottom
        button_frame = ttk.Frame(config_frame)
        button_frame.pack(fill=tk.X, pady=(20, 0))

        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(button_frame, variable=self.progress_var, mode='determinate')
        self.progress_bar.pack(fill=tk.X, pady=(0, 10))

        # Status and buttons
        status_button_frame = ttk.Frame(button_frame)
        status_button_frame.pack(fill=tk.X)

        self.status_label = ttk.Label(status_button_frame, text="Aguardando configuração", style="Status.TLabel")
        self.status_label.pack(side=tk.LEFT, padx=5)

        ttk.Button(status_button_frame, text="Executar Solução", style="Success.TButton",
                   command=self.testar_solucao).pack(side=tk.RIGHT, padx=5)
        ttk.Button(status_button_frame, text="Treinar Agente", style="Primary.TButton",
                   command=self.treinar_agente).pack(side=tk.RIGHT, padx=5)

        # ================ RESULTS TAB ================
        # Improved results view with solution details
        result_top_frame = ttk.Frame(result_frame)
        result_top_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(result_top_frame, text="Solução Encontrada", style="Header.TLabel").pack(side=tk.LEFT)
        self.custo_total_label = ttk.Label(result_top_frame, text="Custo Total: 0.00", style="Header.TLabel")
        self.custo_total_label.pack(side=tk.RIGHT)

        # Tree view in a frame with scrollbars
        tree_frame = ttk.Frame(result_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # More detailed columns for the solution
        self.tree = ttk.Treeview(tree_frame, columns=('Passo', 'Ação', 'Custo', 'Estado'), show='headings')
        self.tree.heading('Passo', text='Passo')
        self.tree.heading('Ação', text='Ação')
        self.tree.heading('Custo', text='Custo')
        self.tree.heading('Estado', text='Estado')

        self.tree.column('Passo', width=60, anchor='center')
        self.tree.column('Ação', width=200)
        self.tree.column('Custo', width=80, anchor='center')
        self.tree.column('Estado', width=500)

        # Add scrollbars
        scroll_y = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=scroll_y.set)

        scroll_x = ttk.Scrollbar(result_frame, orient=tk.HORIZONTAL, command=self.tree.xview)
        scroll_x.pack(fill=tk.X, padx=10)
        self.tree.configure(xscrollcommand=scroll_x.set)

        # ================ GRAPHS TAB ================
        # Configure the graph tab with multiple plot options
        graph_top_frame = ttk.Frame(graph_frame)
        graph_top_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(graph_top_frame, text="Análise de Desempenho", style="Header.TLabel").pack(side=tk.LEFT)

        # Graph type selector
        self.graph_type = tk.StringVar(value="rewards")
        graph_selector_frame = ttk.Frame(graph_top_frame)
        graph_selector_frame.pack(side=tk.RIGHT)

        ttk.Radiobutton(graph_selector_frame, text="Recompensas", variable=self.graph_type,
                        value="rewards", command=self.update_graphs).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(graph_selector_frame, text="Passos", variable=self.graph_type,
                        value="steps", command=self.update_graphs).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(graph_selector_frame, text="Exploração", variable=self.graph_type,
                        value="exploration", command=self.update_graphs).pack(side=tk.LEFT, padx=5)

        # The actual graph canvas
        self.fig = plt.Figure(figsize=(10, 5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # Data storage for plotting
        self.rewards_history = []
        self.steps_history = []
        self.epsilon_history = []

    def treinar_agente(self):
        """Handle agent training with progress updates"""
        try:
            estado_inicial = Estado(
                self.pos_macaco.get(),
                self.pos_cadeira.get(),
                self.pos_vara.get(),
                self.pos_bananas.get(),
                False, False, False
            )

            self.status_label.config(text="Treinando agente...")
            self.root.update_idletasks()

            # Reset progress bar
            self.progress_var.set(0)
            self.progress_bar.start()

            start_time = time.time()

            self.agente = MonkeyQLearning(
                estado_inicial,
                alpha=self.alpha.get(),
                gamma=self.gamma.get(),
                epsilon=self.epsilon.get()
            )

            # Train with progress callback
            episodios_completos, recompensas, passos = self.agente.treinar(
                num_episodios=self.num_episodios.get(),
                progress_callback=self.update_progress
            )

            duracao = time.time() - start_time
            self.progress_bar.stop()

            # Store data for plotting
            self.rewards_history = recompensas
            self.steps_history = passos
            self.epsilon_history = [max(self.agente.min_epsilon,
                                        self.agente.epsilon * (self.agente.epsilon_decay ** i))
                                    for i in range(len(passos))]

            taxa_sucesso = len(episodios_completos) / self.num_episodios.get() * 100
            self.status_label.config(
                text=f"Treinamento concluído - Sucesso: {taxa_sucesso:.1f}% em {duracao:.1f}s"
            )

            # Update graphs
            self.update_graphs()

            messagebox.showinfo(
                "Treinamento Concluído",
                f"Treinamento completado em {duracao:.1f} segundos.\n"
                f"Taxa de sucesso: {taxa_sucesso:.1f}%\n"
                f"Veja os gráficos para mais detalhes."
            )

        except Exception as e:
            messagebox.showerror("Erro no Treinamento", f"Ocorreu um erro: {str(e)}")
            self.status_label.config(text="Erro durante o treinamento")
            self.progress_bar.stop()

    def update_progress(self, value):
        """Update progress bar during training"""
        self.progress_var.set(value)
        self.root.update_idletasks()

    def testar_solucao(self):
        """Test the learned policy and display results"""
        if not self.agente:
            messagebox.showwarning("Aviso", "É necessário treinar o agente primeiro!")
            return

        # Clear previous results
        for item in self.tree.get_children():
            self.tree.delete(item)

        estado_inicial = Estado(
            self.pos_macaco.get(),
            self.pos_cadeira.get(),
            self.pos_vara.get(),
            self.pos_bananas.get(),
            False, False, False
        )

        self.status_label.config(text="Executando solução...")
        self.root.update_idletasks()

        solucao = self.agente.executar_politica(estado_inicial)

        if solucao:
            custo_total = 0
            for i, (acao, estado, custo) in enumerate(solucao, 1):
                self.tree.insert('', 'end', values=(i, acao, f"{custo:.2f}", str(estado)))
                custo_total += custo

            self.custo_total_label.config(text=f"Custo Total: {custo_total:.2f}")
            self.status_label.config(text=f"Solução encontrada em {len(solucao)} passos")

            # Auto-scroll to the last action
            self.tree.see(self.tree.get_children()[-1])

            messagebox.showinfo("Solução Encontrada",
                                f"Solução encontrada em {len(solucao)} passos\nCusto total: {custo_total:.2f}")
        else:
            self.custo_total_label.config(text="Custo Total: N/A")
            self.status_label.config(text="Solução não encontrada")
            messagebox.showwarning("Aviso", "Solução não encontrada!")

    def update_graphs(self):
        """Update the graphs based on current selection"""
        if not self.rewards_history:
            return

        self.fig.clear()

        if self.graph_type.get() == "rewards":
            self.plot_rewards()
        elif self.graph_type.get() == "steps":
            self.plot_steps()
        else:
            self.plot_exploration()

        self.canvas.draw()

    def plot_rewards(self):
        """Plot rewards over episodes"""
        ax = self.fig.add_subplot(111)
        ax.plot(self.rewards_history, 'b-', label='Recompensa por Episódio')

        # Add moving average
        window_size = min(100, len(self.rewards_history) // 10)
        if window_size > 1:
            moving_avg = np.convolve(self.rewards_history, np.ones(window_size) / window_size, mode='valid')
            ax.plot(range(window_size - 1, len(self.rewards_history)), moving_avg,
                    'r-', linewidth=2, label=f'Média Móvel ({window_size} episódios)')

        ax.set_title('Recompensas por Episódio', fontsize=12)
        ax.set_xlabel('Episódio', fontsize=10)
        ax.set_ylabel('Recompensa Total', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()

    def plot_steps(self):
        """Plot steps per episode"""
        ax = self.fig.add_subplot(111)
        ax.plot(self.steps_history, 'g-', label='Passos por Episódio')

        # Add success markers
        success_episodes = [i for i, steps in enumerate(self.steps_history) if steps < 100]
        ax.plot(success_episodes, [self.steps_history[i] for i in success_episodes],
                'ro', markersize=4, label='Episódios de Sucesso')

        ax.set_title('Passos por Episódio', fontsize=12)
        ax.set_xlabel('Episódio', fontsize=10)
        ax.set_ylabel('Número de Passos', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()

    def plot_exploration(self):
        """Plot exploration rate decay"""
        ax = self.fig.add_subplot(111)
        ax.plot(self.epsilon_history, 'm-', label='Taxa de Exploração (ε)')
        ax.set_title('Decaimento da Taxa de Exploração', fontsize=12)
        ax.set_xlabel('Episódio', fontsize=10)
        ax.set_ylabel('Valor de ε', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()