import tkinter as tk
from tkinter import ttk, messagebox
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
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

    def treinar(self, num_episodios=1000, max_steps=100, verbose=False):
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


class MonkeyBananaGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Q-Learning: Problema do Macaco e as Bananas")
        self.root.geometry("1000x700")

        self.pos_macaco = tk.StringVar(value="canto1")
        self.pos_cadeira = tk.StringVar(value="canto3")
        self.pos_vara = tk.StringVar(value="canto3")
        self.pos_bananas = tk.StringVar(value="canto3")

        self.num_episodios = tk.IntVar(value=1000)
        self.alpha = tk.DoubleVar(value=0.1)
        self.gamma = tk.DoubleVar(value=0.9)
        self.epsilon = tk.DoubleVar(value=0.3)

        self.agente = None

        notebook = ttk.Notebook(root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        config_frame = ttk.Frame(notebook)
        notebook.add(config_frame, text="Configuração")

        result_frame = ttk.Frame(notebook)
        notebook.add(result_frame, text="Resultados")

        graph_frame = ttk.Frame(notebook)
        notebook.add(graph_frame, text="Gráficos")

        estado_frame = ttk.LabelFrame(config_frame, text="Configuração Inicial")
        estado_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(estado_frame, text="Posição do Macaco:").grid(row=0, column=0, padx=5, pady=5)
        ttk.Combobox(estado_frame, textvariable=self.pos_macaco, values=positions).grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(estado_frame, text="Posição da Cadeira:").grid(row=1, column=0, padx=5, pady=5)
        ttk.Combobox(estado_frame, textvariable=self.pos_cadeira, values=positions).grid(row=1, column=1, padx=5,
                                                                                         pady=5)

        ttk.Label(estado_frame, text="Posição da Vara:").grid(row=2, column=0, padx=5, pady=5)
        ttk.Combobox(estado_frame, textvariable=self.pos_vara, values=positions).grid(row=2, column=1, padx=5, pady=5)

        ttk.Label(estado_frame, text="Posição das Bananas:").grid(row=3, column=0, padx=5, pady=5)
        ttk.Combobox(estado_frame, textvariable=self.pos_bananas, values=positions).grid(row=3, column=1, padx=5,
                                                                                         pady=5)

        params_frame = ttk.LabelFrame(config_frame, text="Parâmetros Q-Learning")
        params_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(params_frame, text="Número de Episódios:").grid(row=0, column=0, padx=5, pady=5)
        ttk.Entry(params_frame, textvariable=self.num_episodios, width=10).grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(params_frame, text="Taxa de Aprendizado (α):").grid(row=1, column=0, padx=5, pady=5)
        ttk.Entry(params_frame, textvariable=self.alpha, width=10).grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(params_frame, text="Fator de Desconto (γ):").grid(row=2, column=0, padx=5, pady=5)
        ttk.Entry(params_frame, textvariable=self.gamma, width=10).grid(row=2, column=1, padx=5, pady=5)

        ttk.Label(params_frame, text="Exploração Inicial (ε):").grid(row=3, column=0, padx=5, pady=5)
        ttk.Entry(params_frame, textvariable=self.epsilon, width=10).grid(row=3, column=1, padx=5, pady=5)

        button_frame = ttk.Frame(config_frame)
        button_frame.pack(fill=tk.X, padx=10, pady=10)

        self.status_label = ttk.Label(button_frame, text="Status: Não treinado")
        self.status_label.pack(side=tk.LEFT, padx=5, pady=5)

        ttk.Button(button_frame, text="Treinar Agente", command=self.treinar_agente).pack(side=tk.RIGHT, padx=5, pady=5)
        ttk.Button(button_frame, text="Testar Solução", command=self.testar_solucao).pack(side=tk.RIGHT, padx=5, pady=5)

        self.tree = ttk.Treeview(result_frame, columns=('Ação', 'Custo', 'Estado'), show='headings')
        self.tree.heading('Ação', text='Ação')
        self.tree.heading('Custo', text='Custo')
        self.tree.heading('Estado', text='Estado')
        self.tree.column('Ação', width=200)
        self.tree.column('Custo', width=80)
        self.tree.column('Estado', width=400)
        self.tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        scrollbar = ttk.Scrollbar(result_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.custo_total_label = ttk.Label(result_frame, text="Custo Total: 0.00")
        self.custo_total_label.pack(anchor='w', padx=15, pady=5)

        self.graph_container = ttk.Frame(graph_frame)
        self.graph_container.pack(fill=tk.BOTH, expand=True)

        self.fig = plt.Figure(figsize=(10, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_container)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def treinar_agente(self):
        try:
            estado_inicial = Estado(
                self.pos_macaco.get(),
                self.pos_cadeira.get(),
                self.pos_vara.get(),
                self.pos_bananas.get(),
                False, False, False
            )

            self.status_label.config(text="Status: Treinando...")
            self.root.update_idletasks()

            start_time = time.time()

            self.agente = MonkeyQLearning(
                estado_inicial,
                alpha=self.alpha.get(),
                gamma=self.gamma.get(),
                epsilon=self.epsilon.get()
            )

            episodios_completos, recompensas, passos = self.agente.treinar(
                num_episodios=self.num_episodios.get(),
                verbose=True
            )

            duracao = time.time() - start_time

            taxa_sucesso = len(episodios_completos) / self.num_episodios.get() * 100
            self.status_label.config(
                text=f"Status: Treinado ({taxa_sucesso:.1f}% sucesso, {duracao:.1f}s)"
            )

            self.atualizar_graficos(recompensas, passos)

            messagebox.showinfo(
                "Treinamento Concluído",
                f"Treinamento completado em {duracao:.1f} segundos.\n"
                f"Taxa de sucesso: {taxa_sucesso:.1f}%\n"
                f"Veja os gráficos para mais detalhes."
            )

        except Exception as e:
            messagebox.showerror("Erro no Treinamento", f"Ocorreu um erro: {str(e)}")
            self.status_label.config(text="Status: Erro no treinamento")

    def testar_solucao(self):
        if not self.agente:
            messagebox.showwarning("Aviso", "É necessário treinar o agente primeiro!")
            return

        for item in self.tree.get_children():
            self.tree.delete(item)

        estado_inicial = Estado(
            self.pos_macaco.get(),
            self.pos_cadeira.get(),
            self.pos_vara.get(),
            self.pos_bananas.get(),
            False, False, False
        )

        solucao = self.agente.executar_politica(estado_inicial)

        if solucao:
            custo_total = 0
            for i, (acao, estado, custo) in enumerate(solucao):
                self.tree.insert('', 'end', values=(acao, f"{custo:.2f}", str(estado)))
                custo_total += custo

            self.custo_total_label.config(text=f"Custo Total: {custo_total:.2f}")
            messagebox.showinfo("Solução Encontrada",
                                f"Solução encontrada em {len(solucao)} passos\nCusto total: {custo_total:.2f}")
        else:
            self.custo_total_label.config(text="Custo Total: N/A")
            messagebox.showwarning("Aviso", "Solução não encontrada!")

    def atualizar_graficos(self, recompensas, passos):
        self.fig.clear()

        ax1 = self.fig.add_subplot(121)
        ax2 = self.fig.add_subplot(122)

        ax1.plot(recompensas, 'b-')
        ax1.set_title('Recompensas por Episódio')
        ax1.set_xlabel('Episódio')
        ax1.set_ylabel('Recompensa Total')

        ax2.plot(passos, 'r-')
        ax2.set_title('Passos por Episódio')
        ax2.set_xlabel('Episódio')
        ax2.set_ylabel('Número de Passos')

        window_size = min(100, len(recompensas))
        if window_size > 0:
            moving_avg = np.convolve(recompensas, np.ones(window_size) / window_size, mode='valid')
            ax1.plot(range(window_size - 1, len(recompensas)), moving_avg, 'g-', linewidth=2, label='Média Móvel')
            ax1.legend()

        self.fig.tight_layout()
        self.canvas.draw()


def main():
    root = tk.Tk()
    app = MonkeyBananaGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()