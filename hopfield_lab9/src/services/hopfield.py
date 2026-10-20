import math
import random
from typing import List, Dict
from src.domain.models import Pattern


class HopfieldNetwork:
    """
    Математическое ядро нейронной сети Хопфилда.
    Обучает матрицу весов W и выполняет ассоциативное распознавание образов.
    """

    def __init__(self, patterns: List[Pattern]):
        self.patterns = patterns
        self.n = len(patterns[0].vector)  # Количество нейронов (49)
        self.weights: List[List[int]] = [[0] * self.n for _ in range(self.n)]
        self._train()

    def _train(self):
        """
        Обучение сети по правилу Хебба (формула 4.1).
        Формирует симметричную матрицу с нулевой главной диагональю (w_ii = 0).
        """
        self.weights = [[0] * self.n for _ in range(self.n)]
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    self.weights[i][j] = sum(p.vector[i] * p.vector[j] for p in self.patterns)

    def theoretical_capacity(self) -> float:
        """Оценка максимальной емкости сети по формуле (4.5)."""
        return round(self.n / (2 * math.log(self.n) + math.log(math.log(self.n))), 2)

    @staticmethod
    def _activate(val: float) -> int:
        """Биполярная пороговая функция активации (формула 4.4)."""
        return 1 if val > 0 else -1

    def add_noise(self, vector: List[int], noise_percent: float) -> List[int]:
        """Инвертирует заданный процент пикселей (нейронов) в образе."""
        noisy = list(vector)
        num_invert = round(self.n * (noise_percent / 100.0))
        indices = random.sample(range(self.n), min(num_invert, self.n))
        for idx in indices:
            noisy[idx] = -noisy[idx]
        return noisy

    def recognize(self, input_vector: List[int], mode: str = "async", max_epochs: int = 40) -> Dict:
        """
        Воспроизведение (распознавание) образа по формуле (4.3).
        Поддерживает асинхронный и синхронный режимы функционирования.
        """
        state = list(input_vector)
        epochs = 0

        for epoch in range(1, max_epochs + 1):
            epochs = epoch
            prev_state = list(state)

            if mode == "sync":
                # Синхронный режим: все нейроны обновляются одновременно
                new_state = [0] * self.n
                for i in range(self.n):
                    net_sum = sum(self.weights[i][j] * prev_state[j] for j in range(self.n))
                    new_state[i] = self._activate(net_sum)
                state = new_state
            else:
                # Асинхронный режим: нейроны обновляются по одному в случайном порядке
                indices = list(range(self.n))
                random.shuffle(indices)
                for i in indices:
                    net_sum = sum(self.weights[i][j] * state[j] for j in range(self.n))
                    state[i] = self._activate(net_sum)

            # Если состояние стабилизировалось (достигнут локальный минимум энергии)
            if state == prev_state:
                break

        # Проверяем, совпал ли итоговый вектор с одним из эталонов
        matched_pattern = None
        for p in self.patterns:
            if state == p.vector:
                matched_pattern = p.name
                break

        # Считаем процент совпадения пикселей с каждым эталоном
        similarities = {}
        for p in self.patterns:
            matches = sum(1 for a, b in zip(state, p.vector) if a == b)
            similarities[p.name] = round((matches / self.n) * 100, 1)

        return {
            "restored_vector": state,
            "matched_name": matched_pattern if matched_pattern else "Не распознан (ложный аттрактор)",
            "is_success": matched_pattern is not None,
            "epochs": epochs,
            "similarities": similarities
        }

    def run_noise_experiment(self, trials_per_level: int = 10, mode: str = "async") -> Dict:
        """
        Выполняет пункт 4 и 5 задания:
        Тестирует сеть на 12 уровнях зашумления (по 10 тестовых образов для каждого класса).
        """
        noise_levels = [10, 20, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]
        results_by_pattern = {p.name: [] for p in self.patterns}
        average_accuracy = []
        samples_table = {p.name: {} for p in self.patterns}

        for noise in noise_levels:
            total_correct = 0
            for p in self.patterns:
                correct_count = 0
                for t in range(trials_per_level):
                    noisy_vec = self.add_noise(p.vector, noise)
                    res = self.recognize(noisy_vec, mode=mode)
                    if res["matched_name"] == p.name:
                        correct_count += 1

                    # Сохраняем первый пример для визуальной Таблицы 4.1
                    if t == 0:
                        samples_table[p.name][noise] = {
                            "noisy": noisy_vec,
                            "restored": res["restored_vector"],
                            "success": res["matched_name"] == p.name
                        }

                acc = round((correct_count / trials_per_level) * 100, 1)
                results_by_pattern[p.name].append(acc)
                total_correct += correct_count

            avg_acc = round((total_correct / (len(self.patterns) * trials_per_level)) * 100, 1)
            average_accuracy.append(avg_acc)

        return {
            "noise_levels": noise_levels,
            "results_by_pattern": results_by_pattern,
            "average_accuracy": average_accuracy,
            "samples_table": samples_table,
            "trials_per_level": trials_per_level
        }