import os
from typing import Any

from config import ACOConfig, ACOConfig_dto
from loader import load_problem
from ant_colony import AntColonyOpt, ACOResult
from plotting import plot_tour, plot_param_influence, plot_convergence
from utils.file_work import save_json, parse_json
from utils.type_fromating import to_builtin

from pathlib import Path
import matplotlib.pyplot as plt
from dataclasses import replace, asdict
from statistics import mean
import argparse
from concurrent.futures import ProcessPoolExecutor

PARAMS_DATA_PATH = "data/params"
OPT_DATA_PATH = "data/opt_params.json"
OPTIMAL = 7542

def print_detailed(results, param_name):
    print(f"\n===== {param_name.upper()} =====")

    for val, data in results.items():
        print(f"\n{param_name} = {val}")
        print(f"Runs: {data['lengths']}")
        print(f"Avg: {data['avg_len']:.2f}")
        print(f"Std: {data['std_len']:.2f}")
        print(f"Avg iter: {data['avg_iter']:.1f}")
        print(f"Success: {data['success_rate']*100:.1f}%")

def async_executor(job: tuple[Any, Any, str, Any, int]) -> dict[str, Any]:
    dist, base_config, param_name, param_value, seed = job

    # Делаем копию конфига и изменяем его
    config = replace(base_config)
    setattr(config, param_name, param_value)
    if param_name == "num_ants":
        config.one_ant_per_vert = False

    config.seed = seed

    aco = AntColonyOpt(dist, config)
    result = aco.solve()

    return {
        "param_value": param_value,
        "seed": seed,
        "best_len": result.best_len,
        "best_iter": result.best_iter,
        "history_best": result.history_best,
    }

def async_experimets(
    dist,
    base_config,
    param_name: str,
    values: list[Any],
    seeds: list[int],
) -> dict[Any, dict[str, list[Any]]]:
    jobs = [
        (dist, base_config, param_name, val, seed)
        for val in values
        for seed in seeds
    ]

    max_workers = min(os.cpu_count() or 1, len(jobs))

    # Создаём заготовку для результатов
    grouped: dict[Any, dict[str, list[Any]]] = {
        val: {"lens": [], "best_iters": [], "histories": []}
        for val in values
    }

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for out in executor.map(async_executor, jobs, chunksize=1):
            bucket = grouped[out["param_value"]]
            bucket["lens"].append(out["best_len"])
            bucket["best_iters"].append(out["best_iter"])
            bucket["histories"].append(out["history_best"])

    return grouped

def async_run_experiment(dist, base_config: ACOConfig_dto, param_name: str, values: list[int] | list[float], seeds: list[int]):
    print(f"\n=== Testing {param_name} ===")
    results = async_experimets(dist, base_config, param_name, values, seeds)

    output = {}
    for val in values:
        lens = results[val]["lens"]
        best_iters = results[val]["best_iters"]
        histories = results[val]["histories"]

        avg_len = mean(lens)
        dev_lens = []
        for len in lens:
            dev_lens.append((len - OPTIMAL) / OPTIMAL * 100)
        dev_len = mean(dev_lens)
        avg_iter = mean(best_iters)

        output[val] = {
            "lens": lens,
            "avg_len": avg_len,
            "dev_len": dev_len,
            "avg_iter": avg_iter,
            "histories": histories
        }

        print(
            f"{param_name}={val} | "
            f"avg_len={avg_len:.2f} | dev_len={dev_len:.2f} | "
            f"avg_iter={avg_iter:.1f}"
        )

    # Сохраним значения
    save_json(f"{PARAMS_DATA_PATH}/{param_name}.json", output)
    return output



def run_experiment(dist, base_config: ACOConfig_dto, param_name: str, values: list[int] | list[float], seeds: list[int]):
    results = {}

    for val in values:
        print(f"\n=== Testing {param_name} = {val} ===")

        lens = []
        best_iters = []
        histories = []


        for seed in seeds:
            config = replace(base_config)
            setattr(config, param_name, val)
            if param_name == "num_ants":
                config.one_ant_per_vert = False
            config.seed = seed

            aco = AntColonyOpt(dist, config)
            result = aco.solve()

            lens.append(result.best_len)
            best_iters.append(result.best_iter)
            histories.append(result.history_best)

        avg_len = mean(lens)
        dev_lens = []
        for len in lens:
            dev_lens.append((len - OPTIMAL) / OPTIMAL * 100)
        dev_len = mean(dev_lens)
        avg_iter = mean(best_iters)
        
        results[val] = {
            "lens": lens,
            "avg_len": avg_len,
            "dev_len": dev_len,
            "avg_iter": avg_iter,
            "histories": histories
        }

        print(
            f"{param_name}={val} | "
            f"avg_len={avg_len:.2f} | dev_len={dev_len:.2f} | "
            f"avg_iter={avg_iter:.1f}"
        )

    # Сохраним значения
    save_json(f"{PARAMS_DATA_PATH}/{param_name}.json", results)
    return results


def main(args: argparse.Namespace) -> None:
    file = Path("berlin52.tsp")
    if not file.exists() or not file.is_file():
        raise ValueError('Файл "berlin52.tsp" не доуступен')
    coords, dist = load_problem(file)

    config = ACOConfig_dto(
        one_ant_per_vert=True,
        num_iters_without_improve=1500,
        max_iter=1500,
        alpha=1.0,
        beta=3.0,
        rho=0.7,
        q=100.0,
        deposit_scheme="all",
        use_elit=False,
        elit_weight=3.0,
        tau_min=1e-8,
        tau_max=1e8,
    )

    seeds = list(range(10))
    experiments = {
        "alpha": [0.5, 1.0, 1.5, 2.0],
        # "beta": [1, 2, 3, 5, 7],
        # "rho": [0.9, 0.8, 0.7, 0.5],
        # "q": [100, 200, 500],
        # "num_ants": [20, 30, 50]
    }

    all_results = {}

    if(args.param_analis):
        for param, values in experiments.items():
            #result = run_experiment(dist, config, param, values, seeds)
            result = async_run_experiment(dist, config, param, values, seeds)
            all_results[param] = result

            plot_param_influence(param, result, OPTIMAL)
    else:
        # Используем уже сохранённые данныые
        data_path = Path(PARAMS_DATA_PATH)
        data_files = list(data_path.glob("*.json"))

        for file in data_files:
            
            result = parse_json(str(file))

            plot_param_influence(file.stem, result, OPTIMAL)
    

    
    histories = []
    result_opt: ACOResult
    if (args.calc_opt):

        aco = AntColonyOpt(dist, config)

        result_opt = aco.solve()

        save_json(OPT_DATA_PATH, to_builtin(asdict(result_opt)))
    else:
        result_dict = parse_json(OPT_DATA_PATH)
        result_opt = ACOResult(**result_dict)

    plot_tour(coords, result_opt.best_cicle, title=f"Наикротчайший маршрут, длина = {result_opt.best_len:.2f}, итерация сход. = {result_opt.best_iter}, всего итераций = {result_opt.num_iter}")

    histories.append(result_opt.history_best)
    plot_convergence(histories)

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--param_analis", action="store_true") # Флаг расчёта данных для анализа влияния параметров
    parser.add_argument("--calc_opt", action="store_true") # Флаг расчёта данных при оптимальных параметрах
    main(parser.parse_args())


"""
config = ACOConfig(
        one_ant_per_vert=True,
        num_iters_without_improve=1500,
        max_iter=1500,
        alpha=1.0,
        beta=3.0,
        rho=0.7,
        q=100.0,
        #seed=42,
        start_city=None,
        deposit_scheme="all",
        use_elit=False,
        elit_weight=3.0,
        tau_min=1e-8,
        tau_max=1e8,
    )
"""