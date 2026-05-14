"""Category definitions for project classification."""

import csv
import os
from typing import Dict


def _load_categories_from_csv() -> Dict[str, dict]:
    """Load categories from CSV file."""
    csv_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "categorias.csv")
    categories = {}

    try:
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                nombre = row["nombre"]
                categories[nombre] = {
                    "id": int(row["id"]),
                    "nombre": nombre,
                    "definicion": row["definicion"],
                }
        return categories
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Categories CSV file not found at {csv_path}. "
            "Make sure the data/categorias.csv file exists."
        )


# Load categories from CSV
DEFINICIONES_DE_CATEGORIAS = _load_categories_from_csv()
