import tkinter as tk
from tkinter import ttk, messagebox
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
from threading import Thread

# Movimentos possíveis no cubo 2x2 (minimal set sufficient to solve)
movimentos = ["F", "F'", "R", "R'", "U", "U'"]

class EstadoCubo:
    def __init__(self, cubo=None, movimentos_feitos=None):
        if cubo is None:
            # Estado inicial (cubo resolvido)
            self.cubo = {
                'F': [['G', 'G'], ['G', 'G']],  # Frente (Green)
                'B': [['B', 'B'], ['B', 'B']],  # Trás (Blue)
                'U': [['W', 'W'], ['W', 'W']],  # Cima (White)
                'D': [['Y', 'Y'], ['Y', 'Y']],  # Baixo (Yellow)
                'L': [['O', 'O'], ['O', 'O']],  # Esquerda (Orange)
                'R': [['R', 'R'], ['R', 'R']]   # Direita (Red)
            }
        else:
            self.cubo = {face: [linha[:] for linha in cubo[face]] for face in cubo}

        self.movimentos_feitos = movimentos_feitos.copy() if movimentos_feitos else []

    def __eq__(self, other):
        if not isinstance(other, EstadoCubo):
            return False
        return all(
            self.cubo[face] == other.cubo[face]
            for face in self.cubo
        )

    def __hash__(self):
        return hash(self.to_tuple())

    def __repr__(self):
        return f"EstadoCubo(movimentos={len(self.movimentos_feitos)})"

    def to_tuple(self):
        """Converte o estado para uma tupla imutável para uso como chave no dicionário Q."""
        return tuple(
            tuple(tuple(linha) for linha in self.cubo[face])
            for face in sorted(self.cubo.keys())
        )

    def copiar(self):
        """Retorna uma cópia profunda do estado."""
        novo_cubo = {face: [linha[:] for linha in self.cubo[face]] for face in self.cubo}
        return EstadoCubo(novo_cubo, self.movimentos_feitos)

    def aplicar_movimento(self, movimento):
        """Aplica um movimento ao cubo e retorna o novo estado."""
        novo_estado = self.copiar()
        face = movimento[0]
        sentido = movimento[1] if len(movimento) > 1 else ''

        # Rotacionar a face principal
        if sentido == "'":  # Anti-horário
            novo_estado.cubo[face] = [
                [self.cubo[face][1][0], self.cubo[face][0][0]],
                [self.cubo[face][1][1], self.cubo[face][0][1]]
            ]
        else:  # Horário
            novo_estado.cubo[face] = [
                [self.cubo[face][0][1], self.cubo[face][1][1]],
                [self.cubo[face][0][0], self.cubo[face][1][0]]
            ]

        # Rotacionar as bordas adjacentes
        if face == 'F':
            u = [self.cubo['U'][1][0], self.cubo['U'][1][1]]
            r = [self.cubo['R'][0][0], self.cubo['R'][1][0]]
            d = [self.cubo['D'][0][0], self.cubo['D'][0][1]]
            l = [self.cubo['L'][0][1], self.cubo['L'][1][1]]
            if sentido == "'":
                novo_estado.cubo['U'][1] = [l[1], l[0]]
                novo_estado.cubo['R'][0][0], novo_estado.cubo['R'][1][0] = u[0], u[1]
                novo_estado.cubo['D'][0] = [r[1], r[0]]
                novo_estado.cubo['L'][0][1], novo_estado.cubo['L'][1][1] = d[0], d[1]
            else:
                novo_estado.cubo['U'][1] = r
                novo_estado.cubo['R'][0][0], novo_estado.cubo['R'][1][0] = d[1], d[0]
                novo_estado.cubo['D'][0] = l
                novo_estado.cubo['L'][0][1], novo_estado.cubo['L'][1][1] = u[1], u[0]
        elif face == 'R':
            u = [self.cubo['U'][0][1], self.cubo['U'][1][1]]
            b = [self.cubo['B'][0][0], self.cubo['B'][1][0]]
            d = [self.cubo['D'][0][1], self.cubo['D'][1][1]]
            f = [self.cubo['F'][0][1], self.cubo['F'][1][1]]
            if sentido == "'":
                novo_estado.cubo['U'][0][1], novo_estado.cubo['U'][1][1] = f[0], f[1]
                novo_estado.cubo['B'][0][0], novo_estado.cubo['B'][1][0] = [u[1], u[0]]
                novo_estado.cubo['D'][0][1], novo_estado.cubo['D'][1][1] = b[0], b[1]
                novo_estado.cubo['F'][0][1], novo_estado.cubo['F'][1][1] = [d[1], d[0]]
            else:
                novo_estado.cubo['U'][0][1], novo_estado.cubo['U'][1][1] = [b[1], b[0]]
                novo_estado.cubo['B'][0][0], novo_estado.cubo['B'][1][0] = d
                novo_estado.cubo['D'][0][1], novo_estado.cubo['D'][1][1] = [f[1], f[0]]
                novo_estado.cubo['F'][0][1], novo_estado.cubo['F'][1][1] = u
        elif face == 'U':
            f = [self.cubo['F'][0][0], self.cubo['F'][0][1]]
            r = [self.cubo['R'][0][0], self.cubo['R'][0][1]]
            b = [self.cubo['B'][0][0], self.cubo['B'][0][1]]
            l = [self.cubo['L'][0][0], self.cubo['L'][0][1]]
            if sentido == "'":
                novo_estado.cubo['F'][0] = r
                novo_estado.cubo['R'][0] = b
                novo_estado.cubo['B'][0] = l
                novo_estado.cubo['L'][0] = f
            else:
                novo_estado.cubo['F'][0] = l
                novo_estado.cubo['R'][0] = f
                novo_estado.cubo['B'][0] = r
                novo_estado.cubo['L'][0] = b

        # Adicionar movimento ao histórico
        novo_estado.movimentos_feitos.append(movimento)
        return novo_estado

    def esta_resolvido(self):
        """Verifica se o cubo está resolvido (todas as faces têm a mesma cor)."""
        for face in self.cubo.values():
            cor = face[0][0]
            for linha in face:
                for cubinho in linha:
                    if cubinho != cor:
                        return False
        return True

    def embaralhar(self, num_movimentos=10):
        """Embaralha o cubo com movimentos aleatórios."""
        estado = self
        for _ in range(num_movimentos):
            movimento = random.choice(movimentos)
            estado = estado.aplicar_movimento(movimento)
        return estado

    def calcular_recompensa(self):
        """Calcula a recompensa com base no quão próximo está de ser resolvido."""
        if self.esta_resolvido():
            return 100
        recompensa = 0
        for face in self.cubo.values():
            cor = face[0][0]
            for linha in face:
                for cubinho in linha:
                    if cubinho == cor:
                        recompensa += 1
        recompensa -= len(self.movimentos_feitos) * 0.1
        return recompensa

def obter_acoes_possiveis(estado, max_movimentos=20):
    """Retorna lista de tuplas com (nome_acao, novo_estado, recompensa)."""
    if len(estado.movimentos_feitos) >= max_movimentos:
        return []
    acoes_possiveis = []
    for movimento in movimentos:
        novo_estado = estado.aplicar_movimento(movimento)
        recompensa = novo_estado.calcular_recompensa()
        acoes_possiveis.append((movimento, novo_estado, recompensa))
    return acoes_possiveis

class CuboQLearning:
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

    def treinar(self, num_episodios=1000, max_steps=50, verbose=False):
        episodios_completos = []
        recompensas_por_episodio = []
        passos_por_episodio = []

        print("\n=== INICIANDO TREINAMENTO ===")
        print(f"Parâmetros: α={self.alpha:.2f}, γ={self.gamma:.2f}, ε={self.epsilon:.2f}")

        for episodio in range(1, num_episodios + 1):
            estado = self.estado_inicial.embaralhar(5)
            recompensa_total = 0
            step = 0

            if verbose or episodio % 100 == 0:
                print(f"\nEpisódio {episodio}/{num_episodios}, ε={self.epsilon:.3f}")

            while step < max_steps and not estado.esta_resolvido():
                resultado_acao = self.choose_action(estado)
                if resultado_acao is None or resultado_acao[0] is None:
                    if verbose:
                        print("  Sem ações possíveis!")
                    break
                nome_acao, proximo_estado, recompensa = resultado_acao
                recompensa = recompensa if recompensa is not None else 0.0
                if verbose or episodio % 100 == 0:
                    print(f"{step + 1}. {nome_acao:5} | Recompensa: {recompensa:7.2f} | Movimentos: {len(proximo_estado.movimentos_feitos)}")
                estado_tuple = estado.to_tuple()
                proximo_estado_tuple = proximo_estado.to_tuple()
                proximas_acoes = obter_acoes_possiveis(proximo_estado)
                self.update_q_value(estado_tuple, nome_acao, recompensa, proximo_estado_tuple, proximas_acoes)
                estado = proximo_estado
                recompensa_total += recompensa
                step += 1

            recompensas_por_episodio.append(recompensa_total)
            passos_por_episodio.append(step)

            if estado.esta_resolvido():
                episodios_completos.append(episodio)
                if verbose or episodio % 100 == 0:
                    print(f"✅ Cubo resolvido em {step} passos! Recompensa total: {recompensa_total:.2f}")

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

    def simplify_moves(self, moves):
        """Simplifica a sequência de movimentos removendo pares que se cancelam."""
        simplified = []
        i = 0
        while i < len(moves):
            if i + 1 < len(moves) and moves[i][0] == moves[i+1][0] and (
                (moves[i].endswith("'") and not moves[i+1].endswith("'")) or
                (not moves[i].endswith("'") and moves[i+1].endswith("'"))
            ):
                i += 2
            else:
                simplified.append(moves[i])
                i += 1
        return simplified

    def executar_politica(self, estado_inicial, max_steps=50):
        estado = estado_inicial
        plano = []

        print("\n=== EXECUTANDO POLÍTICA APRENDIDA ===")

        for step in range(max_steps):
            if estado.esta_resolvido():
                simplified_moves = self.simplify_moves(estado.movimentos_feitos)
                print(f"✅ Cubo resolvido em {step} passos!")
                print(f"Sequência de movimentos: {' '.join(simplified_moves)}")
                break
            acoes_possiveis = obter_acoes_possiveis(estado)
            if not acoes_possiveis:
                print("❌ Sem ações possíveis!")
                break
            estado_tuple = estado.to_tuple()
            q_values = [(self.get_q_value(estado_tuple, acao[0]), acao) for acao in acoes_possiveis]
            max_q, (nome_acao, proximo_estado, recompensa) = max(q_values, key=lambda x: x[0])
            recompensa = recompensa if recompensa is not None else 0.0
            recompensa_str = f"{recompensa:.2f}"
            plano.append((nome_acao, proximo_estado, recompensa))
            print(f"{step + 1}. {nome_acao:5} | Recompensa: {recompensa_str} | Movimentos: {len(proximo_estado.movimentos_feitos)}")
            estado = proximo_estado

        if not estado.esta_resolvido():
            print(f"❌ Política não conseguiu resolver o cubo em {max_steps} passos")
            return None

        return plano

class CuboMagicoGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Q-Learning: Cubo Mágico 2x2")
        self.root.geometry("1000x700")

        self.num_episodios = tk.IntVar(value=1000)
        self.alpha = tk.DoubleVar(value=0.1)
        self.gamma = tk.DoubleVar(value=0.9)
        self.epsilon = tk.DoubleVar(value=0.3)
        self.num_embaralhamentos = tk.IntVar(value=5)

        self.agente = None

        notebook = ttk.Notebook(root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        config_frame = ttk.Frame(notebook)
        notebook.add(config_frame, text="Configuração")

        result_frame = ttk.Frame(notebook)
        notebook.add(result_frame, text="Resultados")

        graph_frame = ttk.Frame(notebook)
        notebook.add(graph_frame, text="Gráficos")

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

        ttk.Label(params_frame, text="Embaralhamentos Iniciais:").grid(row=4, column=0, padx=5, pady=5)
        ttk.Entry(params_frame, textvariable=self.num_embaralhamentos, width=10).grid(row=4, column=1, padx=5, pady=5)

        button_frame = ttk.Frame(config_frame)
        button_frame.pack(fill=tk.X, padx=10, pady=10)

        self.status_label = ttk.Label(button_frame, text="Status: Não treinado")
        self.status_label.pack(side=tk.LEFT, padx=5, pady=5)

        ttk.Button(button_frame, text="Treinar Agente", command=self.treinar_agente).pack(side=tk.RIGHT, padx=5, pady=5)
        ttk.Button(button_frame, text="Testar Solução", command=self.testar_solucao).pack(side=tk.RIGHT, padx=5, pady=5)

        self.tree = ttk.Treeview(result_frame, columns=('Passo', 'Movimento', 'Recompensa', 'Total Movimentos'), show='headings')
        self.tree.heading('Passo', text='Passo')
        self.tree.heading('Movimento', text='Movimento')
        self.tree.heading('Recompensa', text='Recompensa')
        self.tree.heading('Total Movimentos', text='Total Movimentos')
        self.tree.column('Passo', width=50)
        self.tree.column('Movimento', width=100)
        self.tree.column('Recompensa', width=100)
        self.tree.column('Total Movimentos', width=100)
        self.tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        scrollbar = ttk.Scrollbar(result_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.solucao_label = ttk.Label(result_frame, text="Solução: Não testada")
        self.solucao_label.pack(anchor='w', padx=15, pady=5)

        self.graph_container = ttk.Frame(graph_frame)
        self.graph_container.pack(fill=tk.BOTH, expand=True)

        self.fig = plt.Figure(figsize=(10, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_container)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def treinar_agente(self):
        def train_in_thread():
            try:
                estado_inicial = EstadoCubo()
                self.status_label.config(text="Status: Treinando...")
                self.root.update_idletasks()

                start_time = time.time()
                self.agente = CuboQLearning(
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
                self.root.after(0, lambda: self.status_label.config(
                    text=f"Status: Treinado ({taxa_sucesso:.1f}% sucesso, {duracao:.1f}s)"
                ))
                self.atualizar_graficos(recompensas, passos)
                self.root.after(0, lambda: messagebox.showinfo(
                    "Treinamento Concluído",
                    f"Treinamento completado em {duracao:.1f} segundos.\n"
                    f"Taxa de sucesso: {taxa_sucesso:.1f}%\n"
                    f"Veja os gráficos para mais detalhes."
                ))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Erro no Treinamento", f"Ocorreu um erro: {str(e)}"))
                self.root.after(0, lambda: self.status_label.config(text="Status: Erro no treinamento"))

        thread = Thread(target=train_in_thread)
        thread.start()

    def testar_solucao(self):
        if not self.agente:
            messagebox.showwarning("Aviso", "É necessário treinar o agente primeiro!")
            return

        for item in self.tree.get_children():
            self.tree.delete(item)

        estado_inicial = EstadoCubo().embaralhar(self.num_embaralhamentos.get())
        solucao = self.agente.executar_politica(estado_inicial)

        if solucao:
            for i, (movimento, estado, recompensa) in enumerate(solucao):
                self.tree.insert('', 'end', values=(
                    i + 1,
                    movimento,
                    f"{recompensa:.2f}",
                    len(estado.movimentos_feitos)
                ))
            simplified_moves = self.agente.simplify_moves(estado.movimentos_feitos)
            self.solucao_label.config(text=f"Solução: {len(simplified_moves)} movimentos: {' '.join(simplified_moves)}")
            messagebox.showinfo("Solução Encontrada", f"Solução encontrada em {len(simplified_moves)} movimentos")
        else:
            self.solucao_label.config(text="Solução: Não encontrada")
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
    app = CuboMagicoGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()