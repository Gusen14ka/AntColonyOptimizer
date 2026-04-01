from __future__ import annotations

from dataclasses import dataclass, replace, asdict
import random
from typing import Callable
import os

import numpy as np

from config import ACOConfig, ACOConfig_dto


@dataclass
class ACOResult:
    best_cicle: list[int]
    best_len: float
    best_iter: int
    history_best: list[float]
    history_mean: list[float]
    num_iter: int

class AntColonyOpt:
    config: ACOConfig
    def __init__(self, dist_mat: np.ndarray, config: ACOConfig_dto):
        self.dist_mat = np.array(dist_mat, dtype=float)
        self.n = self.dist_mat.shape[0]
        self.config = self._validate_and_apply_config(config)
        self.rng = np.random.default_rng(self.config.seed)
        self.pheromone = self._init_pheromone()

        if self.dist_mat.shape[0] != self.dist_mat.shape[1]:
            raise ValueError("Матрица расстояний должна быть квадратной.")
        
        self.eta = np.divide(1.0, self.dist_mat, where=self.dist_mat > 0)

        self.rng = np.random.default_rng(self.config.seed)

    def _init_pheromone(self) -> np.ndarray:
        positive = self.dist_mat[self.dist_mat > 0]
        avg_dist = float(np.mean(positive)) if positive.size else 1.0

        tau0 = self.config.init_pheromone
        if tau0 is None or tau0 == 0:
            tau0 = 1.0 / (self.n * avg_dist)

        pheromone = np.full((self.n, self.n), tau0, dtype=float)
        np.fill_diagonal(pheromone, 0.0)
        return pheromone

    def _validate_and_apply_config(self, config: ACOConfig_dto) -> ACOConfig:
  
        if config.one_ant_per_vert:
            if config.num_ants is not None and config.num_ants != self.n:
                raise ValueError("When one_ant_per_vert=True, num_ants must be None or equal to number of cities")
            else:
                num_ants = self.n
        else:
            if config.num_ants is None:
                raise ValueError("num_ants must be set when one_ant_per_vert=False")
            if config.num_ants <= 0:
                raise ValueError("num_ants must be positive")
            num_ants = config.num_ants

        one_ant_per_vert = config.one_ant_per_vert
            
        if config.num_iters_without_improve <= 0:
            raise ValueError("num_iters_without_improve must be positive")
        else:
            num_iters_without_improve = config.num_iters_without_improve

        if config.max_iter <= 0:
            raise ValueError("max_iter must be positive")
        elif config.max_iter < config.num_iters_without_improve:
            max_iter = config.num_iters_without_improve * 2
        else:
            max_iter = config.max_iter

        seed = int(random.randint(0, 2**32 - 1)) if config.seed is None else config.seed

        if config.alpha < 0:
            raise ValueError("alpha must be non-negative")
        else:
            alpha = config.alpha

        if config.beta < 0:
            raise ValueError("beta must be non-negative")
        else:
            beta = config.beta

        if not (0.0 < config.rho < 1.0):
            raise ValueError("rho must be in (0, 1)")
        else:
            rho = config.rho

        if config.q <= 0:
            raise ValueError("q must be positive")
        else:
            q = config.q

        if config.init_pheromone is None:
            init_pheromone = 0
        elif config.init_pheromone <= 0:
            raise ValueError("init_pheromone must be positive")
        else:
            init_pheromone = config.init_pheromone

        if config.deposit_scheme not in {"all", "iteration_best", "global_best"}:
            raise ValueError("Invalid deposit_scheme")
        else:
            deposit_scheme = config.deposit_scheme

        use_elit = config.use_elit

        if config.elit_weight <= 0:
            raise ValueError("elit_weight must be positive")
        else:
            elit_weight = config.elit_weight

        if config.tau_min is not None and config.tau_max is not None:
            if config.tau_min <= 0:
                raise ValueError("tau_min must be positive")
            if config.tau_max <= 0:
                raise ValueError("tau_max must be positive")
            if config.tau_min > config.tau_max:
                raise ValueError("tau_min must be <= tau_max")
            tau_min = config.tau_min
            tau_max = config.tau_max
        else:
            tau_min = 0.0
            tau_max = float("inf")

        

        return ACOConfig(
            num_ants=num_ants,
            one_ant_per_vert=one_ant_per_vert,
            num_iters_without_improve=num_iters_without_improve,
            max_iter=max_iter,
            seed=seed,
            alpha=alpha,
            beta=beta,
            rho=rho,
            q=q,
            init_pheromone=init_pheromone,
            deposit_scheme=deposit_scheme,
            use_elit=use_elit,
            elit_weight=elit_weight,
            tau_max=tau_max,
            tau_min=tau_min
        )

    def _cicle_len(self, cicle: list[int]) -> float:
        length = 0.0
        for i in range(len(cicle) - 1):
            length += self.dist_mat[cicle[i], cicle[i + 1]]
        length += self.dist_mat[cicle[-1], cicle[0]]
        return float(length)

    def _choose_next_vert(
        self,
        current: int,
        unvisited: np.ndarray,
        alpha: float,
        beta: float,
    ) -> int:

        # Создаём массивы привлекательнсти по феромонам и обратной длины
        tau = self.pheromone[current, unvisited]
        eta = self.eta[current, unvisited]

        scores = np.power(tau, alpha) * np.power(eta, beta)
        sum_score = float(np.sum(scores))

        # Проверяем полученную сумму, если какая лажа - выбираем на рандом чтобы не ломать итерации
        if sum_score <= 0.0 or not np.isfinite(sum_score):
            print("[WARN][AntColonyOpt:_choose_next_vert][Некорректная сумма]")
            return int(self.rng.choice(unvisited))
        
        probs = scores / sum_score
        return int(self.rng.choice(unvisited, p=probs))
    
    def _build_cicle(
        self,
        start_vert: int,
        alpha: float,
        beta: float,
    ) -> list[int]:
        visited = np.zeros(self.n, dtype=bool)
        cicle = [start_vert]
        visited[start_vert] = True

        current = start_vert
        for _ in range(self.n - 1):
            unvisited = np.flatnonzero(~visited)
            next_vert = self._choose_next_vert(current, unvisited, alpha, beta)
            cicle.append(next_vert)
            visited[next_vert] = True
            current = next_vert

        return cicle

    def _deposite(self, cicle: list[int], length: float, multipl: float = 1.0) -> None:
        delta = self.config.q * multipl / length

        for i in range(self.n - 1):
            a = cicle[i]
            b = cicle[i + 1]
            self.pheromone[a, b] += delta
            self.pheromone[b, a] += delta

        self.pheromone[cicle[-1], cicle[0]] += delta

    def _evaporate(self, rho: float) -> None:
        self.pheromone *= rho

    def _apply_bounds(self, tau_min: float | None, tau_max: float | None) -> None:
        if tau_min is not None:
            self.pheromone = np.maximum(self.pheromone, tau_min)
        if tau_max is not None:
            self.pheromone = np.minimum(self.pheromone, tau_max)

        np.fill_diagonal(self.pheromone, 0.0)


    def solve(self, on_iteration: Callable[[int, float, float], None] | None = None) -> ACOResult:
        best_cicle: list[int] | None = None
        best_len = float("inf")
        best_iter = 0

        history_best: list[float] = []
        history_mean: list[float] = []

        if self.config.one_ant_per_vert:
            start_cities = np.arange(self.n, dtype=int)
        else:
            start_cities = np.random.randint(0, self.n, self.config.num_ants)

        counter = 0
        iter = 1
        while counter <= self.config.num_iters_without_improve and iter <= self.config.max_iter:
            had_improved = False
            cicles: list[list[int]] = []
            lengths: list[float] = []

            assert self.config.num_ants is not None # Этот азерт не сработает тк конфиг валидируется при загрузке
            seed = self.config.seed

            for i in range(int(self.config.num_ants)):
                start_city = start_cities[i]
                cicle = self._build_cicle(start_city, self.config.alpha, self.config.beta)
                length = self._cicle_len(cicle)
                cicles.append(cicle)
                lengths.append(length)

                if length < best_len:
                    had_improved = True
                    best_len = length
                    best_iter = iter
                    best_cicle = cicle.copy()
                    counter = 0

            # Преобразуем в ndarray массивы, для удобного исследования
            # Изначально используем list тк его append O(1)
            lengths_arr = np.array(lengths, dtype=float)
            iter_best_len_idx = int(np.argmin(lengths))
            iter_best_cicle = cicles[iter_best_len_idx]
            iter_best_len = float(lengths[iter_best_len_idx])

            # Сохраняем данные для анализа
            history_best.append(iter_best_len)
            history_mean.append(float(np.mean(lengths_arr)))

            iter += 1
            if not had_improved:
                counter += 1
                
            # Испаряем феромон
            self._evaporate(self.config.rho)

            # Применяем стратегию изменения феромона
            if self.config.deposit_scheme == "all":
                for cicle, length in zip(cicles, lengths_arr):
                    self._deposite(cicle, float(length))

            elif self.config.deposit_scheme == "iteration_best":
                self._deposite(iter_best_cicle, iter_best_len)

            elif self.config.deposit_scheme == "global_best":
                assert best_cicle is not None
                self._deposite(best_cicle, best_len)
            
            else:
                raise ValueError(
                    'deposit_scheme должен быть одним из: "all", "iteration_best", "global_best".'
                )
            
            if self.config.use_elit and best_cicle is not None:
                self._deposite(best_cicle, best_len, self.config.elit_weight)

            self._apply_bounds(self.config.tau_min, self.config.tau_max)

            if on_iteration is not None:
                on_iteration(iter, best_len, float(np.mean(lengths_arr)))

        if best_cicle is None:
            raise RuntimeError(
                'По окончанию solve best_cicle is None'
            )
        return ACOResult(
            best_cicle, best_len, best_iter, history_best, history_mean, iter
        )
            


        

