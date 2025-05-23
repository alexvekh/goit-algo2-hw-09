import random
import math
import numpy as np
import matplotlib.pyplot as plt

# Визначення функції Сфери
def sphere_function(x):
  return sum(xi ** 2 for xi in x)

def get_random_point(bounds):
    x = random.uniform(bounds[0][0], bounds[0][1])
    y = random.uniform(bounds[1][0], bounds[1][1])
    return (x, y)

def get_custom_point(bounds):
    return (-5.0, 5.0) # (0.0, 0.0)

def clamp(value, min_val, max_val):
    return max(min_val, min(value, max_val))

# Hill Climbing
def hill_climbing(func, bounds, iterations=1000, epsilon=1e-6):
    starting_point = get_custom_point(bounds)   #  get_random_point(bounds)
    print("starting point", starting_point)
    current_point = starting_point
    current_value = func(current_point)

    step_size=0.1
    for iteration in range(iterations):
        x, y = current_point
        neighbors = [
            (clamp(x + step_size, bounds[0][0], bounds[0][1]), y),
            (clamp(x - step_size, bounds[0][0], bounds[0][1]), y),
            (x, clamp(y + step_size, bounds[1][0], bounds[1][1])),
            (x, clamp(y - step_size, bounds[1][0], bounds[1][1]))
        ]

        # Пошук найкращого сусіда
        next_point = None
        next_value = np.inf

        for neighbor in neighbors:
            value = func(neighbor)
            if value < next_value:
                next_point = neighbor
                next_value = value

        if next_point is None or abs(next_value - current_value) < epsilon or euclidean_distance(next_point, current_point) < epsilon:
            print("break")
            break

        # Переходимо до кращого сусіда
        current_point, current_value = next_point, next_value

    return current_point, current_value

def euclidean_distance(p1, p2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

# Random Local Search
def random_local_search(func, bounds, iterations=1000, epsilon=1e-6):
    probability = 0.2
    starting_point = get_custom_point(bounds)   #  get_random_point(bounds)
    current_point = starting_point
    current_value = func(current_point)

    step_size=0.1
    for iteration in range(iterations):
        x, y = current_point
        new_x = x + random.uniform(-step_size, step_size)
        new_y = y + random.uniform(-step_size, step_size)
        new_x = clamp(new_x, bounds[0][0], bounds[0][1])
        new_y = clamp(new_y, bounds[1][0], bounds[1][1])
        new_point = (new_x, new_y)
        new_value = func(new_point)

        if abs(new_value - current_value) < epsilon or euclidean_distance(new_point, current_point) < epsilon:
            print("break")
            break

        # Перевірка умови переходу
        if new_value < current_value or random.random() < probability:
            current_point, current_value = new_point, new_value

    return current_point, current_value


# Simulated Annealing
def simulated_annealing(func, bounds, iterations=1000, temp=1000, cooling_rate=0.95, epsilon=1e-6):

    def simulate(initial_solution, temp, cooling_rate):

        def generate_neighbor(solution):
            x, y = solution
            new_x = x + random.uniform(-1, 1)
            new_y = y + random.uniform(-1, 1)
            new_x = clamp(new_x, bounds[0][0], bounds[0][1])
            new_y = clamp(new_y, bounds[1][0], bounds[1][1])
            return (new_x, new_y)

        current_solution = initial_solution
        current_energy = func(current_solution)

        while temp > epsilon:
            new_solution = generate_neighbor(current_solution)
            new_energy = func(new_solution)
            delta_energy = new_energy - current_energy

            if delta_energy < 0 or random.random() < math.exp(-delta_energy / temp):
                current_solution = new_solution
                current_energy = new_energy

            temp *= cooling_rate

        return current_solution, current_energy

    initial_solution = get_custom_point(bounds)   #  get_random_point(bounds)

    best_solution = None
    best_energy = float("inf")

    for i in range(iterations):
        # print(f"Запуск #{i + 1}")
        solution, energy = simulate(initial_solution, temp, cooling_rate)
        # print(f"Рішення: {solution}, Енергія: {energy}")
        if energy < best_energy:
            best_solution = solution
            best_energy = energy

    # print("\nНайкраще знайдене рішення:")
    # print(f"Рішення: {best_solution}, Енергія: {best_energy}")
    return best_solution, func(best_solution)


if __name__ == "__main__":
  # Межі для функції
  bounds = [(-5, 5), (-5, 5)]

  # Виконання алгоритмів
  print("Hill Climbing:")
  hc_solution, hc_value = hill_climbing(sphere_function, bounds)
  print("Розв'язок:", hc_solution, "Значення:", hc_value)

  print("\nRandom Local Search:")
  rls_solution, rls_value = random_local_search(sphere_function, bounds)
  print("Розв'язок:", rls_solution, "Значення:", rls_value)

  print("\nSimulated Annealing:")
  sa_solution, sa_value = simulated_annealing(sphere_function, bounds)
  print("Розв'язок:", sa_solution, "Значення:", sa_value)

  best_point = min([hc_solution, rls_solution, sa_solution], key=sphere_function)

  # x = np.linspace(bounds[0][0], bounds[0][1], 100)
  # y = np.linspace(bounds[1][0], bounds[1][1], 100)

  # Прибилиженян центру сфери  
  x = np.linspace(-1, 1, 100)
  y = np.linspace(-1, 1, 100)

  X, Y = np.meshgrid(x, y)
  Z = np.array([[sphere_function((i, j)) for i, j in zip(row_x, row_y)] for row_x, row_y in zip(X, Y)])

  plt.figure(figsize=(10, 6))
  plt.contourf(X, Y, Z, levels=50, cmap='viridis')
  plt.colorbar()

  plt.scatter(*hc_solution, color='red', marker='h', label='Hill Climbing')
  plt.scatter(*rls_solution, color='blue', marker='^', label='Random Local Search')
  plt.scatter(*sa_solution, color='green', marker='s', label='Simulated Annealing')

  # plt.scatter(*best_point, color='red', marker='o', label='Найкраща знайдена точка')
  plt.xlabel('x')
  plt.ylabel('y')
  plt.title('Прибилижений центр сфери')
  plt.legend()
  plt.show()
