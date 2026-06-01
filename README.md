# Deepfake Benchmark Framework

Автоматизированный фреймворк для генерации датасетов дипфейков с контролируемой сложностью и воспроизводимой оценки детекторов.

## Возможности

- **Генерация** — создание фейков через FaceFusion с 7 swap-моделями, 2 face enhancer и 4 пресетами сложности (easy / medium / hard / ultra_hard)
- **Кастомные пресеты** — произвольная смесь моделей с весами и 6 параметров постобработки через YAML
- **Честная оценка** — identity-aware split, CelebA-HQ как targets и real одновременно, устранение dataset bias
- **Метрики при дисбалансе** — AUC-ROC, MCC, Balanced Accuracy, авто-подбор порога (без sklearn)
- **Расширяемость** — подключение своего детектора через `DetectorRegistry` за 3 шага, ядро не меняется
- **Три режима** — `generate` (только датасет), `evaluate` (только метрики, FaceFusion не нужен), `full` (всё вместе)

## Быстрый старт

```bash
# Установка
git clone https://github.com/EugenePWN/deepfake-benchmark-framework
cd deepfake-benchmark-framework
poetry install

# Проверить конфиг
poetry run python -m deepfake_benchmark.run --config configs/smoke_test.yaml --dry_run

# Запустить
poetry run python -m deepfake_benchmark.run --config configs/smoke_test.yaml
```

Подробнее → [docs/QUICKSTART.md](docs/QUICKSTART.md)

## Документация

| Документ | Описание |
|---|---|
| [Quick Start](docs/QUICKSTART.md) | Установка, первый запуск, три режима работы |
| [Reference](docs/REFERENCE.md) | Все поля конфига, пресеты, метрики, DetectorRegistry, troubleshooting |


## Структура проекта

```
deepfake_benchmark/
├── config.py              # pydantic v2 конфигурация
├── benchmark.py           # оркестратор (generate / evaluate / full)
├── run.py                 # CLI точка входа
├── types.py               # SampleItem, DetectionResult
├── core/
│   ├── dataset_loader.py
│   ├── dataset_generator.py
│   ├── detector_manager.py
│   ├── metric_evaluator.py
│   ├── reporter.py
│   └── detectors/
│       └── base_detector.py
└── utils/
    ├── dataset_build.py
    └── detect_dataset.py
```

