# Deepfake Benchmark — Quick Start

## Что это

Бенчмарк автоматически генерирует датасеты дипфейков через FaceFusion,
обучает и оценивает детекторы в едином пайплайне.
Всё управляется одним YAML-конфигом.

---

## 1. Установка

```bash
# Клонировать репозиторий
git clone <repo_url>
cd deepfake-bench

# Установить зависимости
poetry install

# Установить FaceFusion (нужен для генерации, не нужен для evaluate)
poetry run python scripts/setup_facefusion.py

# Создать conda-окружение для FaceFusion
conda create -n facefusion python=3.10
conda activate facefusion
pip install -r facefusion/requirements.txt
```

Бенчмарк автоматически найдёт FaceFusion в `<project_root>/facefusion/`
и conda-окружение с именем `facefusion`. Ручная настройка путей не нужна.

---

## 2. Подготовка данных

### Вариант А — свои данные (рекомендуется)

Положи изображения в правильную структуру:

```
datasets/
  celeba_hq/
    images/
      00001.jpg
      00002.jpg
      ...
  vggface2/           # доноры лиц для face-swap
    n000001/
      0001_01.jpg
      ...
```

### Вариант Б — авто-скачивание

```yaml
data:
  auto_download: true
  max_items_per_dataset: 500
```

Поддерживаемые датасеты: `celeba_hq`, `ffhq`, `lfw`, `utkface`.

---

## 3. Первый запуск — smoke-тест

Проверяет что весь пайплайн работает от начала до конца.

```bash
# Создать шаблон конфига
poetry run python -m deepfake_benchmark.run --init smoke_test

# Отредактировать пути в smoke_test.yaml

# Проверить конфиг (без запуска FaceFusion)
poetry run python -m deepfake_benchmark.run --config smoke_test.yaml --dry_run

# Запустить
poetry run python -m deepfake_benchmark.run --config smoke_test.yaml
```

Dry-run покажет:
- найден ли FaceFusion (`[OK]` или `[NOT FOUND]`)
- найдены ли веса детектора (`[OK]` или `[MISSING]`)
- существуют ли папки eval-датасетов (`[OK]`, `[pending]` для mode=full, `[MISSING]` для evaluate)

Результат smoke-теста: `data/smoke_test/results/eval/` — metrics.json, report.html, графики.

---

## 4. Три режима работы

### generate — создать датасет

```yaml
mode: generate

data:
  real_data_root: "datasets"
  source_datasets:
    - celeba_hq
  external_source_dir: "datasets/vggface2"

generation:
  preset: medium
  max_pairs: 1500

output:
  structure: train_val_test
  fake_data_root: "data/fakes"
```

```bash
poetry run python -m deepfake_benchmark.run --config generate.yaml
```

Результат:
```
data/fakes/medium/inswapper_128/*.jpg     # сырые фейки
data/fakes/medium/simswap_256/*.jpg
data/deepfake_dataset/
  train/ real/ fake/                      # собранный датасет
  val/   real/ fake/
  test/  real/ fake/
  manifest.json
```

### evaluate — оценить готовый детектор

FaceFusion **не нужен**. Нужны только папки с данными и `.pth` файл.

```yaml
mode: evaluate

evaluation:
  detectors:
    - name: xception
      architecture: xception
      weights_path: "checkpoints/xception_best.pth"
      threshold: 0.700
  eval_datasets:
    - name: my_test
      real_dir: "data/deepfake_dataset/test/real"
      fake_dir: "data/deepfake_dataset/test/fake"
  output_format: all
```

```bash
poetry run python -m deepfake_benchmark.run --config evaluate.yaml
```

### full — генерация + оценка за один запуск

```yaml
mode: full
# секции data + generation + evaluation
```

---

## 5. Обучение детектора

После `mode: generate` запускай обучение отдельным скриптом. Скрипты обучения детекторов расположены в папке `scripts`

```bash
# Xception (с нуля)
python scripts/train_xception.py --data_dir data/deepfake_dataset --epochs 60 --amp

# EfficientNet-B4 (ImageNet pretrained, двухфазное)
python scripts/train_efficientnet.py --data_dir data/deepfake_dataset --epochs 45 --amp

# F3Net (с нуля, warmup 5 эпох)
python scripts/train_f3net.py --data_dir data/deepfake_dataset --epochs 50 --amp
```

Результат: `checkpoints_*/best.pth` + `results.json` (с optimal_threshold).

---

## 6. Оценка через CLI (без YAML)

```bash
poetry run python -m deepfake_benchmark.utils.detect_dataset \
    --real_dir data/deepfake_dataset/test/real \
    --fake_dir data/deepfake_dataset/test/fake \
    --checkpoints \
        xception:checkpoints/xception_best.pth:0.700 \
        efficientnet:checkpoints/effnet_best.pth:0.588 \
        f3net:checkpoints/f3net_best.pth:0.516
```

---

## 7. Типичные проблемы

**FaceFusion не найден:**
```
[!] FaceFusion not found at: /project/facefusion
    Fix: poetry run python scripts/setup_facefusion.py
```

**Ошибка при загрузке checkpoint:**
```
RuntimeError: Error(s) in loading state_dict
```
Проверь что `architecture:` в конфиге совпадает с моделью checkpoint.


**Папки eval_datasets не существуют в mode=full:**
Это нормально. Dry-run покажет `[pending]` — папки создадутся на шаге generate.

---

## Шпаргалка команд

```bash
# Создать шаблон конфига
poetry run python -m deepfake_benchmark.run --init smoke_test
poetry run python -m deepfake_benchmark.run --init generate
poetry run python -m deepfake_benchmark.run --init evaluate

# Проверить конфиг
poetry run python -m deepfake_benchmark.run --config my.yaml --dry_run

# Запустить
poetry run python -m deepfake_benchmark.run --config my.yaml

# Переопределить устройство
poetry run python -m deepfake_benchmark.run --config my.yaml --device cpu

# Оценить через CLI
poetry run python -m deepfake_benchmark.utils.detect_dataset \
    --real_dir ... --fake_dir ... --checkpoints name:path:threshold
```
