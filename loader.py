from pathlib import Path
import numpy as np
import tsplib95

def load_tsplib_coords(path: Path) -> np.ndarray:
    problem = tsplib95.load(str(path))
    coords_dict = problem.node_coords

    if not coords_dict:
        raise ValueError("В файле TSPLIB не найдены координаты узлов.")
    
    order = sorted(coords_dict.keys()) # pyright: ignore[reportAttributeAccessIssue]
    coords = np.array([coords_dict[i] for i in order], dtype=float) # type: ignore
    return coords

"""
По массиву координат создаёт матрицу расстояний используя стандарт tsplib
dist = floor(sqrt + 0.5)
"""
def build_dist_matrix(coords: np.ndarray) -> np.ndarray:
    diff = coords[:, None, :] - coords[None, :, :] # В одну строчку реализуем логику diff[i][j] = coord[i] - coords[j]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    dist = np.floor(dist + 0.5)
    np.fill_diagonal(dist, 0.0)
    return dist


# Загружает задачу из файла. Возвращает массив координат городов и матрицу расстояний (матрицу графа)
def load_problem(path: Path) -> tuple[np.ndarray, np.ndarray]:
    coords = load_tsplib_coords(path)
    dist = build_dist_matrix(coords)
    return coords, dist