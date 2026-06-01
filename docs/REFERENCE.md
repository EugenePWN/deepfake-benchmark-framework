# Deepfake Benchmark — Полный справочник

## Содержание

1. [Архитектура](#1-архитектура)
2. [YAML-конфиг: все поля](#2-yaml-конфиг)
3. [Пресеты генерации](#3-пресеты-генерации)
4. [PostProcessConfig](#4-postprocessconfig)
5. [Нативные параметры FaceFusion](#5-нативные-параметры-facefusion)
6. [Детекторы и DetectorRegistry](#6-детекторы)
7. [Метрики](#7-метрики)
8. [Утилиты: dataset_build и detect_dataset](#8-утилиты)
9. [Наследование конфигов (extends)](#9-наследование-конфигов)
10. [Авто-определение FaceFusion](#10-авто-определение-facefusion)
11. [Авто-скачивание датасетов](#11-авто-скачивание-датасетов)
12. [Батчевый запуск](#12-батчевый-запуск)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. Архитектура

```
run.py (CLI)                          ← точка входа, YAML → BenchmarkConfig
  │
  ▼
benchmark.py                          ← оркестратор: generate / evaluate / full
  │
  ├─► core/dataset_loader.py          ← загрузка реальных изображений
  ├─► core/dataset_generator.py       ← генерация фейков через FaceFusion
  ├─► utils/dataset_build.py          ← сборка train/val/test + manifest.json
  ├─► core/detector_manager.py        ← инференс детекторов (DetectorRegistry)
  ├─► core/metric_evaluator.py        ← AUC, MCC, BalAcc, F1 (без sklearn)
  └─► core/reporter.py                ← JSON / CSV / HTML-отчёты + графики

config.py                             ← pydantic v2, валидация всех полей
types.py                              ← SampleItem, DetectionResult
```

---

## 2. YAML-конфиг

### Корневые поля

| Поле | Тип | По умолчанию | Описание |
|------|-----|:------------:|----------|
| `mode` | string | `generate` | `generate` / `evaluate` / `full` |
| `device` | string | `cuda` | `cuda` / `cpu` |
| `seed` | int | `42` | Единый seed для всех операций |
| `splits` | dict | `{train: 0.70, val: 0.15, test: 0.15}` | Разбивка. Сумма = 1.0 |

### Секция `data` → LoaderConfig

| Поле | Тип | По умолчанию | Описание |
|------|-----|:------------:|----------|
| `real_data_root` | path | `data/real` | Корневая папка датасетов |
| `source_datasets` | list | `[]` | Имена: `celeba_hq`, `ffhq`, `lfw`, `utkface`, `vggface2`, `custom` |
| `external_source_dir` | path | null | Доноры лиц (VGGFace2). Если null — доноры из source_datasets |
| `max_items_per_dataset` | int | null | Лимит изображений на датасет. null = все |
| `identity_split` | bool | `true` | Разбивать по идентичностям |
| `auto_download` | bool | `false` | Скачивать датасет с HuggingFace если не найден локально |

Структура папок: `<real_data_root>/<dataset_name>/images/*.jpg`

### Секция `generation` → GeneratorConfig

| Поле | Тип | По умолчанию | Описание |
|------|-----|:------------:|----------|
| `preset` | string | `default` | `easy` / `medium` / `hard` / `ultra_hard` / `default` |
| `max_pairs` | int | null | Лимит пар source→target. null = все возможные |
| `output_structure` | string | `flat` | `flat` / `preset_model` |
| `facefusion_dir` | path | null | Путь к FaceFusion. null = авто-определение |
| `facefusion_python` | path | null | Python для FaceFusion. null = авто-определение |
| `skip_existing` | bool | `true` | Пропускать уже сгенерированные файлы |
| `subprocess_timeout` | int | `300` | Таймаут одного вызова FaceFusion (сек) |
| `parallel` | bool | `false` | Параллельная генерация (только CPU) |
| `parallel_workers` | int | `2` | Число воркеров при parallel=true |
| `pairing_mode` | string | `one_for_all` | `one_for_all` / `all_vs_all` / `external_sources` |
| `pairing_seed` | int | `42` | Seed перемешивания пар |
| `identity_aware_pairing` | bool | `true` | Исключать пары одной идентичности |
| `native_args` | dict | `{execution_provider: cuda}` | Параметры FaceFusion напрямую |

### Секция `output` → OutputConfig

| Поле | Тип | По умолчанию | Описание |
|------|-----|:------------:|----------|
| `structure` | string | `train_val_test` | `train_val_test` / `flat_eval` |
| `fake_data_root` | path | `data/fakes` | Куда сохранять фейки |
| `results_root` | path | `data/results` | Куда сохранять отчёты |
| `skip_dataset_build` | bool | `false` | true = только генерация, без сборки train/val/test |
| `output_dir` | path | null | Только для `flat_eval`: путь к плоскому датасету |
| `n_per_class` | int | null | Только для `flat_eval`: лимит на класс |
| `copy_real` | bool | `true` | Копировать real/ при `flat_eval` |
| `real_source` | string | `targets` | `targets` / `loader` — откуда брать real/ |

### Секция `evaluation` → EvalConfig

| Поле | Тип | По умолчанию | Описание |
|------|-----|:------------:|----------|
| `output_format` | string | `json` | `json` / `csv` / `html` / `all` |
| `save_plots` | bool | `true` | Сохранять PNG-графики |
| `threshold_metric` | string | `f1` | Метрика авто-подбора порога: `f1` / `balanced_acc` / `mcc` |

#### `evaluation.detectors[i]` → DetectorEntryConfig

| Поле | Тип | По умолчанию | Описание |
|------|-----|:------------:|----------|
| `name` | string | обязательно | Имя в отчёте |
| `architecture` | string | обязательно | `xception` / `efficientnet` / `f3net` |
| `weights_path` | path | обязательно | Путь к `.pth` |
| `threshold` | float | `0.5` | Порог P(fake). 0.5 = авто-подбор |
| `img_size` | int | null | Входной размер. null = из архитектуры |
| `batch_size` | int | `16` | Батч при инференсе |

#### `evaluation.eval_datasets[i]` → EvalDatasetConfig

| Поле | Тип | По умолчанию | Описание |
|------|-----|:------------:|----------|
| `name` | string | обязательно | Имя в отчёте |
| `real_dir` | path | обязательно | Папка с реальными |
| `fake_dir` | path | обязательно | Папка с фейками |
| `n_per_class` | int | null | Лимит на класс. null = все |
| `mix_with` | list | `[]` | Имена датасетов для объединения |

---

## 3. Пресеты генерации

### easy — явные артефакты
- Модели: inswapper_128 (100%)
- Маска: box, blur=0.1
- Enhancer: нет
- Пост: без деградации

### medium — социальные сети
- Модели: inswapper_128 (60%) + simswap_256 (40%)
- Маска: box+occlusion, blur=0.3, padding=[3,3,3,3]
- Enhancer: нет
- Пост: JPEG 75–82, color_jitter=0.15, resize=0.85

### hard — мессенджеры
- Модели: inswapper_128_fp16 (40%) + simswap_512 (35%) + ghost_256 (25%)
- Маска: box+occlusion+region, blur=0.6, padding=[5,5,5,5]
- Enhancer: GFPGAN 1.4 (blend=80)
- Пост: JPEG 60–72, resize=0.6–0.75, blur=1–2, noise=4–6, jitter=0.2–0.3, crop=0.92

### ultra_hard — максимальное качество
- Модели: inswapper_128_fp16 (100%)
- Маска: region+occlusion, blur=0.6, padding=[4,4,4,4]
- Enhancer: CodeFormer (blend=88)
- Пост: JPEG 94, color_jitter=0.03

### Кастомный пресет

```yaml
generation:
  preset: my_preset
  custom_presets:
    my_preset:
      model_mix:
        - model_name: inswapper_128
          weight: 0.7
          post_process:
            jpeg_quality: 85
        - model_name: simswap_256
          weight: 0.3
          post_process:
            jpeg_quality: 75
            resize_factor: 0.85
      face_mask_types: [box, occlusion]
      face_mask_blur: 0.3
```

---

## 4. PostProcessConfig

Шесть параметров, применяются после FaceFusion.
Порядок фиксирован: **crop → resize → blur → color jitter → noise → JPEG**.

| Параметр | Диапазон | Описание |
|----------|:--------:|----------|
| `random_crop_ratio` | 0.5–1.0 | Кроп центра. 1.0 = без кропа |
| `resize_factor` | 0–1.0 | Ресайз вниз + обратно. null = нет |
| `gaussian_blur_r` | 0–20 | Радиус Гауссова размытия. 0 = нет |
| `color_jitter` | 0–1.0 | ±factor для яркости/контраста/насыщенности |
| `add_noise_std` | 0–50 | σ аддитивного Гауссова шума |
| `jpeg_quality` | 1–95 | JPEG-сжатие. null = без сжатия |

JPEG последним — как в реальной цепочке обработки.

---

## 5. Нативные параметры FaceFusion

Передаются в CLI FaceFusion через `generation.native_args`:

```yaml
generation:
  native_args:
    execution_provider: cuda
    face_detector_model: retinaface
    face_detector_score: 0.7
    face_selector_mode: one
    face_mask_types: [box, occlusion]
    face_mask_blur: 0.3
    face_mask_padding: [0, 0, 0, 0]
    face_enhancer_model: gfpgan_1.4
    face_enhancer_blend: 80
```

Доступные swap-модели:

| Модель | Разрешение | Особенности |
|--------|:----------:|-------------|
| `inswapper_128` | 128×128 | Базовая, быстрая |
| `inswapper_128_fp16` | 128×128 | FP16 |
| `simswap_256` | 256×256 | Сохраняет атрибуты |
| `simswap_512_unofficial` | 512×512 | Высокое разрешение |
| `ghost_256_unet_1/2/3` | 256×256 | Ghost-архитектура |
| `blendface_256` | 256×256 | Мягкое блендирование |
| `uniface_256` | 256×256 | Обобщение |

Face enhancer:

| Модель | Когда |
|--------|-------|
| null | easy — хотим видеть артефакты |
| `gfpgan_1.4` | hard — улучшает текстуры |
| `codeformer` | ultra_hard — лучшее восстановление |

---

## 6. Детекторы и DetectorRegistry

### Встроенные архитектуры

| Архитектура | Вход | Нормализация | Pretrained |
|-------------|:----:|:------------:|:----------:|
| `xception` | 299×299 | [0.5] | нет |
| `efficientnet` | 380×380 | ImageNet | да |
| `f3net` | 299×299 | [0.5] | нет |

### Подключение своего детектора

```python
# deepfake_benchmark/core/detectors/my_detector.py

from ..base_detector import BaseDetector, DetectionResult
from ...types import SampleItem

@DetectorRegistry.register
class MyDetector(BaseDetector):
    name = "my_model"

    def load(self, weights_path, **kwargs):
        self.model = torch.load(weights_path)
        self._loaded = True

    def predict_one(self, item: SampleItem) -> DetectionResult:
        img = preprocess(item.media_path)
        score = self.model(img)
        return self._make_result(item, score)
```

Три шага: унаследовать, реализовать `name` + `load()` + `predict_one()`, добавить `@register`.

---

## 7. Метрики

Все реализованы без sklearn.

| Метрика | Threshold-free | Назначение |
|---------|:--------------:|-----------|
| AUC-ROC | ✅ | Основная метрика сравнения |
| AUC-PR | ✅ | Чувствительна к дисбалансу |
| MCC | ❌ | Лучшая одиночная метрика при дисбалансе |
| Balanced Accuracy | ❌ | Среднее recall по классам |
| F1 | ❌ | Гармоническое среднее precision и recall |
| Sensitivity (TPR) | ❌ | Доля найденных фейков |
| Specificity (TNR) | ❌ | Доля правильно распознанных реальных |
| Accuracy | ❌ | Присутствует, но **не является основной** при дисбалансе |

Оптимальный порог: grid search [0.2, 0.7] с шагом 1/50, максимизируя F1 на val.

---

## 8. Утилиты

### dataset_build.py

Собирает train/val/test из сырых фейков.

```bash
poetry run python -m deepfake_benchmark.utils.dataset_build \
    --real_dir datasets/celeba_hq/images \
    --fakes_dir data/fakes \
    --out_dir data/deepfake_dataset \
    --presets easy medium hard \
    --restrict_real_to_used_targets
```

Ключевая фича: identity-aware split — фейк попадает в тот же сплит что target.

### detect_dataset.py

Прогон детекторов без полного пайплайна.

```bash
# Через manifest.json
poetry run python -m deepfake_benchmark.utils.detect_dataset \
    --dataset_dir data/deepfake_dataset \
    --split test \
    --checkpoints xception:checkpoints/xception_best.pth:0.700

# Через папки real/ + fake/
poetry run python -m deepfake_benchmark.utils.detect_dataset \
    --real_dir data/test/real \
    --fake_dir data/test/fake \
    --checkpoints \
        xception:checkpoints/xception_best.pth:0.700 \
        efficientnet:checkpoints/effnet_best.pth:0.588
```

Формат `--checkpoints`: `name:path:threshold` (threshold опционален, по умолчанию 0.5).

---

## 9. Наследование конфигов

```yaml
# configs/base.yaml
device: cuda
seed: 42
data:
  real_data_root: "datasets"
  source_datasets: [celeba_hq]
  external_source_dir: "datasets/vggface2"

# configs/experiments/hard_test.yaml
extends: ../base.yaml          # наследуем всё из base.yaml
mode: generate
generation:
  preset: hard                 # переопределяем только это
  max_pairs: 300
```

Слияние рекурсивное: вложенные словари мержатся, списки перезаписываются.
Циклические ссылки обнаруживаются автоматически.

---

## 10. Авто-определение FaceFusion

Если `facefusion_dir` и `facefusion_python` не указаны — бенчмарк ищет сам.

**Папка FaceFusion** (в порядке приоритета):
1. `<project_root>/facefusion/` — после `setup_facefusion.py`
2. `~/facefusion/`
3. `./facefusion/`

**Python для FaceFusion** (conda-окружения):
- Имена: `facefusion`, `ff`, `deepfake`, `facefusion_env`
- Пути: `~/miniconda3/envs/`, `~/anaconda3/envs/`, `C:/ProgramData/miniconda3/envs/`
- Fallback: `sys.executable`

Dry-run показывает что найдено:
```
facefusion_dir: /project/facefusion  OK  [auto-detected]
python        : ~/conda/envs/facefusion/python.exe  [auto-detected]
```

---

## 11. Авто-скачивание датасетов

```yaml
data:
  auto_download: true
  source_datasets:
    - celeba_hq
  max_items_per_dataset: 500    # ограничить объём
```

| Датасет | Источник | Зависимость |
|---------|---------|-------------|
| `celeba_hq` | HuggingFace `korexyz/celeba-hq-256x256` | `pip install datasets` |
| `ffhq` | HuggingFace `bitmind/ffhq-256` | `pip install datasets` |
| `utkface` | HuggingFace `Subh775/UTKFace_demographics_V1` | `pip install datasets` |
| `lfw` | sklearn `fetch_lfw_people` | `pip install scikit-learn` |

Датасеты скачиваются в `<real_data_root>/<name>/images/`.
Если папка уже содержит изображения — скачивание пропускается.

---

## 12. Батчевый запуск

### Несколько датасетов за один запуск

```yaml
evaluation:
  eval_datasets:
    - name: facefusion_easy
      real_dir: "data/eval_easy/real"
      fake_dir: "data/eval_easy/fake"
    - name: facefusion_hard
      real_dir: "data/eval_hard/real"
      fake_dir: "data/eval_hard/fake"
    - name: celeb_df_crosstest
      real_dir: "cross_eval/celeb_df/real"
      fake_dir: "cross_eval/celeb_df/fake"
      n_per_class: 300
```

Все датасеты обрабатываются последовательно, результаты в одном отчёте.

### Объединение датасетов (mix_with)

```yaml
eval_datasets:
  - name: facefusion_easy
    real_dir: "data/eval_easy/real"
    fake_dir: "data/eval_easy/fake"
    mix_with: [facefusion_hard]     # смешать в один набор

  - name: facefusion_hard
    real_dir: "data/eval_hard/real"
    fake_dir: "data/eval_hard/fake"
```

Результат: один отчёт `facefusion_easy+facefusion_hard`.

### Через shell-скрипт

```bash
for preset in easy medium hard ultra_hard; do
    poetry run python -m deepfake_benchmark.run \
        --config "configs/eval_${preset}.yaml"
done
```

---

## 13. Troubleshooting

### FaceFusion не находит лицо

```
[Worker] FaceFusion failed: face not detected
```

Решение: уменьши `face_detector_score`:
```yaml
generation:
  native_args:
    face_detector_score: 0.5
```

### Генерация зависает

```
[Worker] timeout (300s)
```

Решение: увеличь таймаут или проверь что модели FaceFusion скачаны:
```yaml
generation:
  subprocess_timeout: 600
```

### CUDA out of memory при инференсе

Решение: уменьши batch_size:
```yaml
evaluation:
  detectors:
    - name: f3net
      batch_size: 8
```

### Ошибка загрузки checkpoint

```
RuntimeError: Error(s) in loading state_dict
```

Проверь совпадение `architecture:` с моделью. Бенчмарк автоматически убирает
префикс `_orig_mod.` от `torch.compile`.

### Оптимальный threshold не найден

Используй `threshold: 0.5` — бенчмарк подберёт автоматически по F1.
Точнее: возьми значение из `results.json` → `optimal_threshold` после обучения.

### Пропущенные зависимости

```bash
# Минимум для mode: evaluate
pip install torch torchvision Pillow numpy pyyaml pydantic

# Для генерации — FaceFusion устанавливается отдельно
poetry run python scripts/setup_facefusion.py

# Для графиков
pip install matplotlib

# Для авто-скачивания датасетов
pip install datasets
```

### Пути не резолвятся правильно

`_resolve_path` ищет в порядке:
1. Абсолютный → как есть
2. `${PROJECT_ROOT}/...` → от корня проекта (pyproject.toml / .git)
3. Существует относительно CWD → используем
4. Иначе → относительно папки конфига

Если конфиг в `configs/`, а пути от корня проекта — всё будет найдено
автоматически через пункт 3 (CWD = корень проекта).
