# Segmentacja wad kabli


## Opis zadania

Zadanie polega na segmentacji binarnej wad na obrazach kabli ze zbioru MVTec Anomaly Detection Dataset. Dla każdego obrazu wejściowego należy wygenerować maskę binarną wskazującą pikselowo obszar wady.

### Dane

- Kategorie wad: bent_wire, cable_swap, combined, cut_inner_insulation, cut_outer_insulation, missing_cable, missing_wire, poke_insulation.
- Obrazy RGB, rozdzielczość 1024×1024 px.
- https://drive.google.com/drive/folders/1bD6UtRyOdgS4jobo-RjU__dCmTFQ_6hi

### Format zgłoszenia

Zgłoszenie (submission) powinno być plikiem ZIP zawierającym:

- model.py — plik z funkcją predict(image)
- requirements.txt — (opcjonalnie) dodatkowe biblioteki Python

```python
def predict(image: np.ndarray) -> np.ndarray:
    """
    Args:
        image: tablica NumPy, kształt (H, W, 3), dtype uint8, RGB

    Returns:
        Maska binarna, kształt (H, W), dtype uint8, wartości 0 lub 255
        255 = wada, 0 = brak wady
    """
```

### Metryka oceny

Wyniki oceniane są metryką IoU (Intersection over Union) uśrednioną po wszystkich obrazach testowych (defective + good). Dla obrazów good maską referencyjną jest maska zerowa.

### Środowisko wykonawcze
- Python 3.12, Docker python:3.12-slim
- Limit czasu: 300 sekund (5 minut)
- Dostępne biblioteki: numpy, Pillow


## How To Run

```bash
uv sync
```

```bash
uv run main.py
```


## TODO
- model :D
