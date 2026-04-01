from dataclasses import dataclass

@dataclass
class ACOConfig_dto:
    num_ants: int | None = None
    one_ant_per_vert: bool = True

    num_iters_without_improve: int = 200
    max_iter: int = 1000
    seed: int | None = None
    
    alpha: float = 1.0
    beta: float = 5
    rho: float = 0.8 # Чтобы не делать каждую итерацию лишнее вычисление 1 - rho, в нашей нотации rho := 1 - rho
    q: float = 100.0

    init_pheromone: float | None = None
    
    # Доп параметры для более умного метода
    # способ выбрать, кто именно кладёт феромон
    deposit_scheme: str = "all" # "all" | "iteration_best" | "global_best"
    
    # Вариация с элитными муравьями
    use_elit: bool = False
    elit_weight: float = 2.0

    # Ограничения для минимального и максимального изменения феромона
    tau_min: float | None = None
    tau_max: float | None = None

@dataclass
class ACOConfig:
    num_ants: int = 52
    one_ant_per_vert: bool = True

    num_iters_without_improve: int = 200
    max_iter: int = 1000
    seed: int = 42
    
    alpha: float = 1.0
    beta: float = 5
    rho: float = 0.8 # Чтобы не делать каждую итерацию лишнее вычисление 1 - rho, в нашей нотации rho := 1 - rho
    q: float = 100.0

    init_pheromone: float = 0.0
    
    # Доп параметры для более умного метода
    # способ выбрать, кто именно кладёт феромон
    deposit_scheme: str = "all" # "all" | "iteration_best" | "global_best"
    
    # Вариация с элитными муравьями
    use_elit: bool = False
    elit_weight: float = 2.0

    # Ограничения для минимального и максимального изменения феромона
    tau_min: float = 0.0
    tau_max: float = float("inf")