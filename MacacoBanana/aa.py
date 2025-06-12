import tkinter as tk
from tkinter import ttk, messagebox
import random
import math
from collections import defaultdict

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

# Custos para ações específicas (agora são recompensas negativas)
ACTION_COSTS = {
    "mover": -1.0,  # Custo base para movimento
    "pegar_vara": -2.0,
    "colocar_vara": -1.5,
    "empurrar_cadeira": -4.0,
    "subir_cadeira": -3.0,
    "apontar_vara": -2.5,
    "agitar_vara": -2.0,
    "objetivo": 100.0  # Recompensa positiva por alcançar o objetivo
}


def dist(a, b):
    """Retorna a distância entre duas posições."""
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


def objetivo(estado):
    """Verifica se o objetivo foi alcançado (macaco tem as bananas)."""
    return estado.tem_bananas


def acoes_disponiveis(estado):
    """Gera as ações possíveis a partir do estado atual."""
    acoes = []

    # Mover-se
    for p in positions:
        if p != estado.pos_macaco:
            acoes.append(("ir_para_" + p, p))

    # Pegar vara do chão
    if estado.pos_macaco == estado.pos_vara and not estado.tem_vara and not estado.vara_na_cadeira and not estado.em_cima:
        acoes.append(("pegar_vara", None))

    # Colocar vara sobre a cadeira
    if estado.tem_vara and estado.pos_macaco == estado.pos_cadeira and not estado.em_cima:
        acoes.append(("colocar_vara_sobre_cadeira", None))

    # Pegar vara em cima da cadeira
    if estado.vara_na_cadeira and estado.pos_macaco == estado.pos_cadeira and estado.em_cima:
        acoes.append(("pegar_vara_em_cima_cadeira", None))

    # Apontar vara para as bananas
    if estado.tem_vara and estado.em_cima and estado.pos_macaco == estado.pos_bananas:
        acoes.append(("apontar_vara", None))

    # Agitar vara (pegar bananas)
    if estado.vara_apontada and estado.tem_vara and estado.em_cima and estado.pos_macaco == estado.pos_bananas:
        acoes.append(("agitar_vara", None))

    # Empurrar cadeira
    if estado.pos_macaco == estado.pos_cadeira and not estado.em_cima:
        for p in positions:
            if p != estado.pos_macaco:
                acoes.append(("empurrar_cadeira_para_" + p, p))

    # Subir na cadeira
    if estado.pos_macaco == estado.pos_cadeira and not estado.em_cima:
        acoes.append(("subir_cadeira", None))

    return acoes


def executar_acao(estado, acao):
    """Executa uma ação e retorna o novo estado e a recompensa."""
    nome_acao, param = acao

    # Criar cópia do estado atual
    novo_estado = Estado(
        estado.pos_macaco, estado.pos_cadeira, estado.pos_vara, estado.pos_bananas,
        estado.tem_vara, estado.em_cima, estado.tem_bananas,
        estado.vara_na_cadeira, estado.vara_apontada
    )

    recompensa = 0

    if nome_acao.startswith("ir_para_"):
        novo_estado.pos_macaco = param
        recompensa = ACTION_COSTS["mover"] * dist(estado.pos_macaco, param)

    elif nome_acao == "pegar_vara":
        novo_estado.tem_vara = True
        recompensa = ACTION_COSTS["pegar_vara"]

    elif nome_acao == "colocar_vara_sobre_cadeira":
        novo_estado.tem_vara = False
        novo_estado.vara_na_cadeira = True
        recompensa = ACTION_COSTS["colocar_vara"]

    elif nome_acao == "pegar_vara_em_cima_cadeira":
        novo_estado.tem_vara = True
        novo_estado.vara_na_cadeira = False
        recompensa = ACTION_COSTS["pegar_vara"]

    elif nome_acao == "apontar_vara":
        novo_estado.vara_apontada = True
        recompensa = ACTION_COSTS["apontar_vara"]

    elif nome_acao == "agitar_vara":
        novo_estado.tem_bananas = True
        recompensa = ACTION_COSTS["objetivo"]  # Grande recompensa por alcançar o objetivo

    elif nome_acao.startswith("empurrar_cadeira_para_"):
        novo_estado.pos_macaco = param
        novo_estado.pos_cadeira = param
        recompensa = ACTION_COSTS["empurrar_cadeira"] + ACTION_COSTS["mover"] * dist(estado.pos_macaco, param)

    elif nome_acao == "subir_cadeira":
        novo_estado.em_cima = True
        recompensa = ACTION_COSTS["subir_cadeira"]

    return novo_estado, recompensa


class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.alpha = alpha  # Taxa de aprendizado
        self.gamma = gamma  # Fator de desconto
        self.epsilon = epsilon  # Taxa de exploração

    def get_q_value(self, estado, acao):
        return self.q_table[hash(estado)][acao]

    def choose_action(self, estado):
        acoes = acoes_disponiveis(estado)
        if not acoes:
            return None

        # Exploração: escolhe ação aleatória
        if random.random() < self.epsilon:
            return random.choice(acoes)

        # Explotação: escolhe ação com maior Q-value
        q_values = [(acao, self.get_q_value(estado, acao)) for acao in acoes]
        max_q = max(q_values, key=lambda x: x[1])[1]
        best_actions = [acao for acao, q in q_values if q == max_q]
        return random.choice(best_actions)

    def learn(self, estado, acao, recompensa, novo_estado):
        acoes_novo_estado = acoes_disponiveis(novo_estado)

        # Valor máximo para o próximo estado
        max_q_novo = max([self.get_q_value(novo_estado, a) for a in acoes_novo_estado]) if acoes_novo_estado else 0

        # Atualização Q-value
        old_q = self.get_q_value(estado, acao)
        new_q = old_q + self.alpha * (recompensa + self.gamma * max_q_novo - old_q)
        self.q_table[hash(estado)][acao] = new_q


def q_learning(estado_inicial, max_episodes=1000, max_steps=100):
    agent = QLearningAgent()

    for episode in range(max_episodes):
        estado = estado_inicial
        total_recompensa = 0

        for step in range(max_steps):
            # Escolher ação
            acao = agent.choose_action(estado)
            if not acao:
                break

            # Executar ação
            novo_estado, recompensa = executar_acao(estado, acao)
            total_recompensa += recompensa

            # Aprender com a experiência
            agent.learn(estado, acao, recompensa, novo_estado)

            # Verificar se alcançou o objetivo
            if objetivo(novo_estado):
                break

            estado = novo_estado

    # Depois do treinamento, executar a política aprendida
    plano = []
    estado = estado_inicial
    custo_total = 0

    for _ in range(max_steps):
        # Escolher a melhor ação (sem explorar)
        agent.epsilon = 0  # Desativa exploração
        acao = agent.choose_action(estado)
        if not acao:
            break

        # Executar ação
        novo_estado, recompensa = executar_acao(estado, acao)
        custo = -recompensa  # Convertemos recompensa negativa em custo positivo

        # Adicionar ao plano
        plano.append((acao[0], novo_estado, custo))
        custo_total += custo

        # Verificar se alcançou o objetivo
        if objetivo(novo_estado):
            break

        estado = novo_estado

    return plano, custo_total


class MonkeyBananaGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Problema do Macaco e as Bananas (Q-Learning)")

        # Variáveis de estado inicial
        self.pos_macaco = tk.StringVar(value="canto1")
        self.pos_cadeira = tk.StringVar(value="canto3")
        self.pos_vara = tk.StringVar(value="canto3")
        self.pos_bananas = tk.StringVar(value="canto3")

        # Frame de configuração
        config_frame = ttk.LabelFrame(root, text="Configuração Inicial")
        config_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # Dropdowns para posições
        ttk.Label(config_frame, text="Posição do Macaco:").grid(row=0, column=0, padx=5, pady=5)
        ttk.Combobox(config_frame, textvariable=self.pos_macaco, values=positions).grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(config_frame, text="Posição da Cadeira:").grid(row=1, column=0, padx=5, pady=5)
        ttk.Combobox(config_frame, textvariable=self.pos_cadeira, values=positions).grid(row=1, column=1, padx=5,
                                                                                         pady=5)

        ttk.Label(config_frame, text="Posição da Vara:").grid(row=2, column=0, padx=5, pady=5)
        ttk.Combobox(config_frame, textvariable=self.pos_vara, values=positions).grid(row=2, column=1, padx=5, pady=5)

        ttk.Label(config_frame, text="Posição das Bananas:").grid(row=3, column=0, padx=5, pady=5)
        ttk.Combobox(config_frame, textvariable=self.pos_bananas, values=positions).grid(row=3, column=1, padx=5,
                                                                                         pady=5)

        # Botão para executar
        ttk.Button(config_frame, text="Executar Solução", command=self.executar_solucao).grid(row=4, column=0,
                                                                                              columnspan=2, pady=10)

        # Frame de resultados
        result_frame = ttk.LabelFrame(root, text="Plano de Ações")
        result_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        # Treeview para mostrar as ações
        self.tree = ttk.Treeview(result_frame, columns=('Ação', 'Custo', 'Estado'), show='headings')
        self.tree.heading('Ação', text='Ação')
        self.tree.heading('Custo', text='Custo')
        self.tree.heading('Estado', text='Estado')
        self.tree.column('Ação', width=200)
        self.tree.column('Custo', width=80)
        self.tree.column('Estado', width=400)
        self.tree.pack(fill=tk.BOTH, expand=True)

        # Barra de rolagem
        scrollbar = ttk.Scrollbar(result_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Label para mostrar custo total
        self.custo_total_label = ttk.Label(result_frame, text="Custo Total: 0.00")
        self.custo_total_label.pack(anchor='w', padx=5, pady=5)

        # Configurar expansão
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)

    def executar_solucao(self):
        # Limpar resultados anteriores
        for item in self.tree.get_children():
            self.tree.delete(item)

        # Criar estado inicial
        estado_inicial = Estado(
            self.pos_macaco.get(),
            self.pos_cadeira.get(),
            self.pos_vara.get(),
            self.pos_bananas.get(),
            False, False, False
        )

        # Executar Q-Learning
        solucao, custo_total = q_learning(estado_inicial)

        if solucao:
            for i, (acao, estado, custo) in enumerate(solucao):
                self.tree.insert('', tk.END, values=(acao, f"{custo:.2f}", str(estado)))

            self.custo_total_label.config(text=f"Custo Total: {custo_total:.2f}")
        else:
            messagebox.showinfo("Sem Solução", "Nenhuma solução foi encontrada para a configuração atual.")
            self.custo_total_label.config(text="Custo Total: 0.00")


# Execução independente para teste
def executar_teste():
    estado_inicial = Estado("canto1", "canto3", "canto3", "canto3", False, False, False)
    solucao, custo_total = q_learning(estado_inicial, max_episodes=500)

    if solucao:
        print("\n=== PLANO DE AÇÕES ===")
        for i, (acao, estado, custo) in enumerate(solucao):
            print(f"{i + 1}. {acao} -> Custo: {custo:.2f} -> {estado}")
        print(f"\nCusto Total da Solução: {custo_total:.2f}")
    else:
        print("Nenhuma solução encontrada.")


# Criar e rodar a interface se for o script principal
if __name__ == "__main__":
    # Executar teste independente para demonstrar funcionamento
    executar_teste()

    # Criar e iniciar interface gráfica
    root = tk.Tk()
    app = MonkeyBananaGUI(root)
    root.mainloop()