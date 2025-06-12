import heapq
import tkinter as tk
from tkinter import scrolledtext

# CLASSE DE ESTADO
class State:
    def __init__(self, left_m, left_c, boat, right_m, right_c, parent=None, action=None, depth=0):
        self.left_m = left_m
        self.left_c = left_c
        self.boat = boat
        self.right_m = right_m
        self.right_c = right_c
        self.parent = parent
        self.action = action
        self.depth = depth  # g(n)
        self.h = None
        self.f = None

    def is_valid(self):
        if self.left_m < 0 or self.left_c < 0 or self.right_m < 0 or self.right_c < 0:
            return False
        if self.left_m > 0 and self.left_m < self.left_c:
            return False
        if self.right_m > 0 and self.right_m < self.right_c:
            return False
        return True

    def is_goal(self):
        return self.left_m == 0 and self.left_c == 0

    def calculate_h(self):
        self.h = (self.left_m + self.left_c + 1) // 2

    def calculate_f(self):
        self.calculate_h()
        self.f = self.depth + self.h

    def __eq__(self, other):
        return (self.left_m == other.left_m and self.left_c == other.left_c and
                self.boat == other.boat and self.right_m == other.right_m and
                self.right_c == other.right_c)

    def __hash__(self):
        return hash((self.left_m, self.left_c, self.boat, self.right_m, self.right_c))

    def __lt__(self, other):
        return self.f < other.f if self.f != other.f else self.h < other.h

    def __str__(self):
        return (f"Left: {self.left_m}M {self.left_c}C | Boat: {self.boat} | "
                f"Right: {self.right_m}M {self.right_c}C | g={self.depth}, h={self.h}, f={self.f}")

# GERA SUCESSORES
def get_successors(parent):
    successors = []
    moves = [(1, 0), (2, 0), (0, 1), (0, 2), (1, 1)]
    direction = 'right' if parent.boat == 'left' else 'left'

    for m, c in moves:
        if parent.boat == 'left':
            new_state = State(parent.left_m - m, parent.left_c - c, direction,
                              parent.right_m + m, parent.right_c + c,
                              parent, f"Levar {m}M e {c}C para direita", parent.depth + 1)
        else:
            new_state = State(parent.left_m + m, parent.left_c + c, direction,
                              parent.right_m - m, parent.right_c - c,
                              parent, f"Trazer {m}M e {c}C para esquerda", parent.depth + 1)

        if new_state.is_valid():
            new_state.calculate_f()
            successors.append(new_state)

    return successors

# FUNÇÃO A*
def a_star_search(initial_state, log_callback):
    open_set = []
    initial_state.calculate_f()
    heapq.heappush(open_set, initial_state)
    closed_set = set()
    g_values = {initial_state: initial_state.depth}
    iteration = 0

    while open_set:
        current = heapq.heappop(open_set)

        log_callback(f"Iteração {iteration}:")
        log_callback(f"Ação: {current.action if current.action else 'Estado inicial'}")
        log_callback(str(current) + "\n")
        iteration += 1

        if current.is_goal():
            return reconstruct_path(current)

        closed_set.add(current)

        for successor in get_successors(current):
            if successor in closed_set:
                continue
            tentative_g = current.depth + 1
            if successor not in g_values or tentative_g < g_values[successor]:
                g_values[successor] = tentative_g
                successor.depth = tentative_g
                successor.calculate_f()
                heapq.heappush(open_set, successor)

    return None

# RECONSTRUIR CAMINHO
def reconstruct_path(state):
    path = []
    while state:
        path.append(state)
        state = state.parent
    return path[::-1]

# MOSTRAR SOLUÇÃO NA INTERFACE
def show_solution(path, log_callback):
    log_callback("\n===== CAMINHO DA SOLUÇÃO FINAL =====\n")
    for i, state in enumerate(path):
        if i == 0:
            log_callback(f"Passo {i}: Estado inicial")
        else:
            log_callback(f"Passo {i}: {state.action}")
        log_callback(str(state) + "\n")

# ======================
# INTERFACE COM TKINTER
# ======================

def iniciar_busca(text_area):
    text_area.delete('1.0', tk.END)

    def log(msg):
        text_area.insert(tk.END, msg + "\n")
        text_area.see(tk.END)

    initial = State(3, 3, 'left', 0, 0)
    solution = a_star_search(initial, log)
    if solution:
        show_solution(solution, log)
    else:
        log("Nenhuma solução encontrada.")

# CONFIGURAÇÃO DA JANELA
def main():
    root = tk.Tk()
    root.title("Missionários e Canibais - A*")

    text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=90, height=30, font=("Courier", 10))
    text_area.pack(padx=10, pady=10)

    btn = tk.Button(root, text="Iniciar Busca A*", command=lambda: iniciar_busca(text_area))
    btn.pack(pady=5)

    root.mainloop()

if __name__ == "__main__":
    main()
