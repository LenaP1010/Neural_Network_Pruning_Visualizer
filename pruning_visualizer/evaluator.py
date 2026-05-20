"""
Модуль для оценки метрик модели: размер, скорость, точность.
"""

import time
import torch
import pandas as pd
from torchvision import datasets, transforms
from tqdm.auto import tqdm

class Evaluator:
    """Класс для оценки метрик модели."""
    
    def __init__(self, device='cpu'):
        self.device = device

    def measure_size(self, model: torch.nn.Module) -> dict:
        """
        Измеряет количество параметров и физический размер модели.
        
        :param model: Модель PyTorch.
        :return: Словарь с результатами.
        """
        parameters = list(model.parameters())
        if not parameters:
            raise ValueError("Модель не содержит параметров для измерения")
        
        num_params = sum(p.numel() for p in parameters)
        size_mb = round(sum(p.element_size() * p.nelement() for p in parameters) / 1e6, 2)
        return {
            "parameters": num_params,
            "size_MB": size_mb
        }

    def measure_inference_speed(self, model: torch.nn.Module, input_shape=(1, 3, 224, 224), iterations=100) -> float:
        """
        Измеряет среднюю скорость инференса модели.
        
        :param model: Модель PyTorch.
        :param input_shape: Форма входного тензора.
        :param iterations: Количество итераций для усреднения.
        :return: Среднее время инференса в миллисекундах.
        """
        model.eval()
        model.to(self.device)
        dummy_input = torch.rand(input_shape).to(self.device)

        # Разогрев
        with torch.no_grad():
            _ = model(dummy_input)

        times = []
        with torch.no_grad():
            for _ in tqdm(range(iterations), desc="Измерение скорости"):
                start = time.time()
                _ = model(dummy_input)
                end = time.time()
                times.append(end - start)

        avg_time_ms = round(sum(times) / len(times) * 1000, 2)
        return avg_time_ms

    def evaluate_accuracy(self, model: torch.nn.Module, dataset_name: str = "cifar10", batch_size: int = 32) -> float:
        """
        Измеряет точность модели на тестовом наборе данных.
        
        :param model: Модель PyTorch.
        :param dataset_name: Имя набора данных (cifar10/mnist).
        :param batch_size: Размер батча.
        :return: Точность в процентах.
        """
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        if dataset_name == "cifar10":
            dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
        elif dataset_name == "mnist":
            dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
        else:
            raise ValueError("Unsupported dataset")

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        model.eval()
        model.to(self.device)

        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Оценка точности"):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = round(correct / total * 100, 2)
        return accuracy

    def generate_report(self, original_model: torch.nn.Module, pruned_model: torch.nn.Module, dataset_name: str = "cifar10") -> pd.DataFrame:
        """
        Генерирует отчёт о сравнении двух моделей.
        
        :param original_model: Исходная модель.
        :param pruned_model: Обрезанная модель.
        :param dataset_name: Имя набора данных (cifar10/mnist).
        :return: Таблица сравнения.
        """
        report = []

        try:
            orig_size = self.measure_size(original_model)
            pruned_size = self.measure_size(pruned_model)

            orig_speed = self.measure_inference_speed(original_model)
            pruned_speed = self.measure_inference_speed(pruned_model)

            orig_acc = self.evaluate_accuracy(original_model, dataset_name)
            pruned_acc = self.evaluate_accuracy(pruned_model, dataset_name)

            report.append({
                "Model": "Original",
                "Parameters": orig_size["parameters"],
                "Size (MB)": orig_size["size_MB"],
                "Inference Speed (ms)": orig_speed,
                "Accuracy (%)": orig_acc
            })

            report.append({
                "Model": "Pruned",
                "Parameters": pruned_size["parameters"],
                "Size (MB)": pruned_size["size_MB"],
                "Inference Speed (ms)": pruned_speed,
                "Accuracy (%)": pruned_acc
            })
        except Exception as e:
            print(f"Ошибка при генерации отчёта: {e}")
            raise

        df = pd.DataFrame(report)
        return df
