import random
import math
import numpy as np
import matplotlib.pyplot as plt

# Визначення функції Сфери
def sphere_function(x):
  return sum(xi ** 2 for xi in x)

def get_rundom_point(bounds):
    x = random.uniform(bounds[0][0], bounds[0][1])
    y = random.uniform(bounds[1][0], bounds[1][1])
    return (x, y)

def get_zero_point(bounds):
    return (0.0, 0.0)

def get_neighbors(current, step_size=0.1):
    x, y = current
    return [
        (x + step_size, y),
        (x - step_size, y),
        (x, y + step_size),
        (x, y - step_size)
    ]

def get_random_neighbor(current, step_size=0.5):
    x, y = current
    new_x = x + random.uniform(-step_size, step_size)
    new_y = y + random.uniform(-step_size, step_size)
    return (new_x, new_y)

# Hill Climbing
def hill_climbing(func, bounds, iterations=1000, epsilon=1e-6):
    starting_point =  (2.0, 2.0)
    current_point = starting_point
    print(current_point)
    current_value = func(current_point)

    for iteration in range(iterations):
        neighbors = get_neighbors(current_point)

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
    starting_point =  (0.0, 0.0)
    starting_point =  (2.0, 2.0)
    current_point = starting_point
    current_value = func(current_point)


    def get_random_neighbor(current, step_size=0.5):
        x, y = current
        new_x = x + random.uniform(-step_size, step_size)
        new_y = y + random.uniform(-step_size, step_size)
        return (new_x, new_y)

    for iteration in range(iterations):
        x, y = current_point
        new_x = x + random.uniform(-step_size, step_size)
        new_y = y + random.uniform(-step_size, step_size)
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
        # def evaluate(solution):
        #     x, y = solution
        #     return (x - 3) ** 2 + (y - 2) ** 2

        def generate_neighbor(solution):
            x, y = solution
            new_x = x + random.uniform(-1, 1)
            new_y = y + random.uniform(-1, 1)
            return (new_x, new_y)

        current_solution = initial_solution
        current_energy = func(current_solution)

        while temp > 0.001:
            new_solution = generate_neighbor(current_solution)
            new_energy = func(new_solution)
            delta_energy = new_energy - current_energy

            if delta_energy < 0 or random.random() < math.exp(-delta_energy / temp):
                current_solution = new_solution
                current_energy = new_energy

            temp *= cooling_rate

        return current_solution, current_energy

    initial_solution = (0, 0)  # Початкова точка
    # temperature = 1000         # Початкова температура
    # cooling_rate = 0.85        # Швидкість охолодження
    # runs = 10                  # Кількість запусків

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


def plot_3d_surface_with_points(bounds, points):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Сітка значень
    x = np.linspace(bounds[0][0], bounds[0][1], 100)
    y = np.linspace(bounds[1][0], bounds[1][1], 100)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[sphere_function((xi, yi)) for xi, yi in zip(row_x, row_y)]
                  for row_x, row_y in zip(X, Y)])

    # Побудова поверхні
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)

    # Точки алгоритмів
    colors = ['red', 'blue', 'green']
    labels = ['Hill Climbing', 'Random Local Search', 'Simulated Annealing']

    for (pt, val), color, label in zip(points, colors, labels):
        ax.scatter(pt[0], pt[1], val, color=color, s=80, label=label)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x, y)')
    ax.set_title('3D Visualization of the Sphere Function')
    ax.legend()
    plt.show()


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


  points = [
        (hc_solution, sphere_function(hc_solution)),
        (rls_solution, sphere_function(rls_solution)),
        (sa_solution, sphere_function(sa_solution))
    ]
  plot_3d_surface_with_points(bounds, points)


  x = np.linspace(bounds[0][0], bounds[0][1], 100)
  y = np.linspace(bounds[1][0], bounds[1][1], 100)
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
  plt.title('Цільова функція та знайдена точка максимуму')
  plt.legend()
  plt.show()
