import torch
import torch.nn as nn
from torch.serialization import safe_globals
import torchvision

class Pruner:
    """Класс для прунинга нейросети."""

    def __init__(self, model: nn.Module):
        self.model = model

    def apply_weight_pruning(self, threshold: float) -> None:
        """Применяет прунинг по абсолютной величине весов."""
        with torch.no_grad():
            for param in self.model.parameters():
                mask = torch.abs(param.data) >= threshold
                param.data *= mask.float()

    @staticmethod

    def load_model(model_path: str) -> nn.Module:
    # Загружаем данные из файла
        loaded_data = torch.load(model_path, map_location='cpu', weights_only=False)

    # Создаём «чистую» архитектуру модели
        model = torchvision.models.resnet18(num_classes=10)

        if isinstance(loaded_data, dict):
        # Если загрузили state_dict (OrderedDict) — используем напрямую
            model.load_state_dict(loaded_data)
        else:
        # Если загрузили полную модель — извлекаем state_dict
            model.load_state_dict(loaded_data.state_dict())

        return model


    def prune_model(model_path: str, output_path: str, method: str, threshold: float) -> None:
        model = Pruner.load_model(model_path)
        pruner = Pruner(model)

        # Считаем нулевые веса до прунинга
        zero_before = sum((p == 0).sum().item() for p in model.parameters())

        if method == "weight":
            pruner.apply_weight_pruning(threshold)

        # Считаем нулевые веса после прунинга
        zero_after = sum((p == 0).sum().item() for p in model.parameters())
        print(f"Нулевые веса: до {zero_before}, после {zero_after}")

        Pruner.save_model(model, output_path)

    @staticmethod
    def save_model(model: nn.Module, path: str) -> None:
        """Сохраняет модель в файл (только state_dict)."""
        torch.save(model.state_dict(), path)
