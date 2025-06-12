import heapq

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
        self.f = None  # f(n) = g(n) + h(n)

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
        people_left = self.left_m + self.left_c
        self.h = (people_left + 1) // 2

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
                f"Right: {self.right_m}M {self.right_c}C | "
                f"g={self.depth}, h={self.h}, f={self.f}")

def get_successors(parent):
    successors = []
    moves = [(1, 0), (2, 0), (0, 1), (0, 2), (1, 1)]

    if parent.boat == 'left':
        for m, c in moves:
            new_state = State(
                parent.left_m - m,
                parent.left_c - c,
                'right',
                parent.right_m + m,
                parent.right_c + c,
                parent,
                f"Levar {m}M e {c}C para direita",
                parent.depth + 1
            )
            if new_state.is_valid():
                new_state.calculate_f()
                successors.append(new_state)
    else:
        for m, c in moves:
            new_state = State(
                parent.left_m + m,
                parent.left_c + c,
                'left',
                parent.right_m - m,
                parent.right_c - c,
                parent,
                f"Trazer {m}M e {c}C para esquerda",
                parent.depth + 1
            )
            if new_state.is_valid():
                new_state.calculate_f()
                successors.append(new_state)

    return successors

def a_star_search(initial_state):
    open_set = []
    initial_state.calculate_f()
    heapq.heappush(open_set, initial_state)
    closed_set = set()
    g_values = {initial_state: initial_state.depth}

    while open_set:
        current = heapq.heappop(open_set)

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

def reconstruct_path(state):
    path = []
    while state:
        path.append(state)
        state = state.parent
    return path[::-1]

def print_solution(path):
    for i, state in enumerate(path):
        if i == 0:
            print(f"Passo {i}: Estado inicial")
        else:
            print(f"Passo {i}: {state.action}")
        print(state)
        print()

# Execução
initial_state = State(3, 3, 'left', 0, 0)
initial_state.calculate_f()
solution = a_star_search(initial_state)

if solution:
    print("Solução encontrada:")
    print_solution(solution)
else:
    print("Nenhuma solução encontrada.")
