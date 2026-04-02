import numpy as np
import matplotlib.pyplot as plt


def plot_tour(coords: np.ndarray, tour: list[int], title: str = "ACO tour") -> None:
    route = tour + [tour[0]]
    path = coords[route]

    plt.figure(figsize=(8, 8))
    plt.plot(path[:, 0], path[:, 1], marker="o")
    plt.scatter(coords[:, 0], coords[:, 1])

    for i, (x, y) in enumerate(coords):
        plt.text(x, y, str(i), fontsize=8, ha="right", va="bottom")

    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.grid(True)


def plot_history(history_best: list[float], history_mean: list[float]) -> None:
    plt.figure(figsize=(9, 5))
    plt.plot(history_best, label="Best so far")
    plt.plot(history_mean, label="Mean ant length")
    plt.title("ACO convergence")
    plt.xlabel("Iteration")
    plt.ylabel("Tour length")
    plt.grid(True)
    plt.legend()

def plot_param_influence(param_name, results, real_answ: int):
    xs = []
    ys = []
    yerr = []

    for val, data in results.items():
        if param_name == "rho":
            xs.append(1 - float(val))
        else:
            xs.append(float(val))
        ys.append(data["avg_len"])
        yerr.append(data["dev_len"])

    
    

    plt.figure()
    plt.errorbar(xs, ys, yerr=yerr, marker='o', label="Решение при заданных параметрах")
    plt.plot(xs, [real_answ] * len(xs), color="green", label="Истинное оптимальное решение")
    plt.title(f"Влиянение параметра {param_name}")
    plt.xlabel(param_name)
    plt.ylabel("Среднее значение длины маршрута")
    plt.grid(True)
    plt.legend()

def plot_convergence(histories, real_answ: int):
    plt.figure()

    for history in histories:
        plt.plot(history, label="Полученное значение")
        plt.plot([real_answ] * len(history), color="green", label="Истинное оптимальное решение")


    plt.title("Сходимость алгоримта при оптимальных значениях")
    plt.xlabel("Итерация")
    plt.ylabel("Лучшая (наименьшая) длина маршрута")
    plt.grid(True)
    plt.legend()