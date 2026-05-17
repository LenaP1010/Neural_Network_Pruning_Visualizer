"""
CLI для инструмента прунинга нейросетей.
"""

import typer
import torch
import torchvision
import matplotlib.pyplot as plt  # Для графиков
import networkx as nx
import seaborn as sns  # Для тепловых карт
from torchviz import make_dot  # Для схемы сети
import numpy as np  # Добавляем импорт numpy

app = typer.Typer()

from pruning_visualizer.pruner import Pruner
from pruning_visualizer.evaluator import Evaluator

# Вспомогательные функции для визуализации

def plot_weight_distribution(model, title):
    """Строит гистограмму распределения весов модели."""
    weights = []
    for param in model.parameters():
        weights.extend(param.detach().numpy().flatten())
    
    plt.hist(weights, bins=100, alpha=0.7, color='blue')
    plt.title(title)
    plt.xlabel('Вес')
    plt.ylabel('Частота')
    plt.show()



def plot_heatmap(model, title):
    """Строит тепловую карту весов модели."""
    weights = []
    for param in model.parameters():
        weights.append(param.detach().numpy())
    
    heatmap = np.concatenate(weights, axis=0)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap, cmap='coolwarm', annot=False)
    plt.title(title)
    plt.show()

def plot_comparison_report(report_df):
    """Строит диаграмму сравнения метрик (размер, скорость, точность)."""
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

    # Столбчатая диаграмма для размеров моделей
    axs[0].bar(report_df['Model'], report_df['Size (MB)'])
    axs[0].set_title('Размеры моделей')
    axs[0].set_ylabel('Размер (MB)')

    # Столбчатая диаграмма для скорости инференса
    axs[1].bar(report_df['Model'], report_df['Inference Speed (ms)'])
    axs[1].set_title('Скорость инференса')
    axs[1].set_ylabel('Время (мс)')

    # Столбчатая диаграмма для точности
    axs[2].bar(report_df['Model'], report_df['Accuracy (%)'])
    axs[2].set_title('Точность')
    axs[2].set_ylabel('Точность (%)')

    plt.tight_layout()
    plt.show()

# Основной CLI-командой для прунинга нейросети

@app.command()
def main(
    model_path: str = typer.Option(..., "-m", "--model-path", help="Путь к файлу модели (.pt, .pth, etc.)"),
    method: str = typer.Option("weight", "-t", "--method", help="Тип прунинга ('weight', 'random', 'structural')"),
    threshold: float = typer.Option(0.1, "-th", "--threshold", help="Порог обрезки (0.0 - 1.0)"),
    visualize: bool = typer.Option(False, "-v", "--visualize", help="Показывать визуализацию изменений"),
):
    """
    Основной CLI-командой для прунинга нейросети.
    
    :param model_path: Путь к файлу модели (.pt, .pth, etc.)
    :param method: Тип прунинга ('weight', 'random', 'structural')
    :param threshold: Порог обрезки (0.0 - 1.0)
    :param visualize: Показывать визуализацию изменений
    """
    output_path = f"{model_path.split('.')[0]}_pruned.pt"
    prune_model(model_path, output_path, method, threshold)
    typer.echo(f"Прунинг завершён. Результат сохранён в {output_path}")



def prune_model(model_path: str, output_path: str, method: str, threshold: float) -> None:
    """Функция для выполнения прунинга модели."""
    # Загружаем модель через Pruner
    model = Pruner.load_model(model_path)
    pruner = Pruner(model)

    if method == "weight":
        pruner.apply_weight_pruning(threshold)
    else:
        raise ValueError(f"Метод '{method}' не поддерживается.")

    Pruner.save_model(model, output_path)
    print(f"Прунинг завершён. Результат сохранён в {output_path}")

# Команда для оценки метрик: размер, скорость, точность

@app.command()
def evaluate_metrics(
    original_model_path: str = typer.Option(..., "-om", "--original-model-path", help="Путь к натренированной модели"),
    pruned_model_path: str = typer.Option(..., "-pm", "--pruned-model-path", help="Путь к обрезанной модели"),
    dataset_name: str = typer.Option("cifar10", "-dn", "--dataset-name", help="Имя набора данных (cifar10/mnist)"),
    device: str = typer.Option("cpu", "-dv", "--device", help="Устройство для оценки (cpu/cuda)")
):
    """
    Команда для оценки метрик: размер, скорость, точность.
    
    :param original_model_path: Путь к натренированной модели.
    :param pruned_model_path: Путь к обрезанной модели.
    :param dataset_name: Имя набора данных (cifar10/mnist).
    :param device: Устройство для оценки (cpu/cuda).
    """
    # Загружаем модели
    original_model = Pruner.load_model(original_model_path)
    pruned_model = Pruner.load_model(pruned_model_path)

    # Визуализация распределения весов
    plot_weight_distribution(original_model, "Распределение весов до прунинга")
    plot_weight_distribution(pruned_model, "Распределение весов после прунинга")


    # Визуализация динамики точности
    thresholds = [0.1, 0.01, 0.001]  # Пример порогов обрезки
    sizes = [44.73, 44.73, 44.73]  # Пример размеров моделей (MB)
    speeds = [10.49, 10.46, 10.50]  # Пример скоростей инференса (мс)
    accuracies = [78.40, 76.84, 76.37]  # Пример значений точности

    def plot_pruning_performance(thresholds, sizes, speeds, accuracies):
        """Строит график взаимосвязи между степенью прунинга и производительностью модели."""
        plt.figure(figsize=(10, 6))
        
        # Степень прунинга (процент оставшихся параметров)
        remaining_params = [(1 - t) * 100 for t in thresholds]
        
        # Строим линии для каждой метрики
        plt.plot(remaining_params, sizes, label='Размер модели (MB)', marker='o')
        plt.plot(remaining_params, speeds, label='Скорость инференса (мс)', marker='o')
        plt.plot(remaining_params, accuracies, label='Точность (%)', marker='o')
        
        plt.title('Взаимосвязь между степенью прунинга и производительностью модели')
        plt.xlabel('Процент оставшихся параметров')
        plt.ylabel('Производительность')
        plt.legend()
        plt.grid(True)
        plt.show()


    # Визуализация динамики точности
    plot_pruning_performance(thresholds, sizes, speeds, accuracies)

    # Создаём оценщик
    evaluator = Evaluator(device=device)

    # Генерируем отчёт
    report = evaluator.generate_report(original_model, pruned_model, dataset_name)

    # Визуализация сравнения метрик
    plot_comparison_report(report)

    # Выводим отчёт
    typer.echo("\nОтчёт о сравнении моделей:\n")
    typer.echo(report.to_string(index=False))

if __name__ == "__main__":
    app()