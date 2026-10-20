from dataclasses import dataclass
from typing import List

@dataclass
class Pattern:
    """Модель биполярного образа для сети Хопфилда."""
    id: str
    name: str
    width: int
    height: int
    vector: List[int]  # Вектор из 1 и -1 длиной width * height


def get_default_patterns() -> List[Pattern]:
    """
    Возвращает 3 эталонных образа (первые буквы фамилии: Б, О, Н)
    размером 7x7 (49 нейронов) с повышенной взаимной ортогональностью,
    чтобы все 3 образа имели устойчивые локальные минимумы в сети Хопфилда.
    """
    raw_grids = {
        "Б": [
            "1111111",
            "1100000",
            "1100000",
            "1111110",
            "1100011",
            "1100011",
            "1111110"
        ],
        "О": [
            "0011100",
            "0100010",
            "1000001",
            "1000001",
            "1000001",
            "0100010",
            "0011100"
        ],
        "Н": [
            "1000001",
            "1000001",
            "1111111",
            "1111111",
            "1111111",
            "1000001",
            "1000001"
        ]
    }

    patterns = []
    for idx, (letter, grid) in enumerate(raw_grids.items(), start=1):
        vector = [1 if char == '1' else -1 for row in grid for char in row]
        patterns.append(Pattern(
            id=str(idx),
            name=f"Буква «{letter}»",
            width=7,
            height=7,
            vector=vector
        ))
    return patterns