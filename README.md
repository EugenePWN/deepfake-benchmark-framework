# Deepfake Benchmark

Локальный бенчмарк для генерации fake-изображений (сейчас основной сценарий: `facefusion`) с запуском через `Poetry` и конфигами YAML.

## Что в проекте уже работает

- запуск пайплайна через `python -m deepfake_benchmark.cli --config ...`
- загрузка изображений из локальных папок `source/target`
- генерация через `FaceFusionGeneratorAdapter`
- режимы паринга:
  - `one_for_all`
  - `all_vs_all`
  - `external_sources`
- конфиги по шаблону `base + overrides` через поле `extends`

## Требования

- Python `3.10+`
- Poetry
- Git (для первичной установки FaceFusion)
- каталог **`facefusion/`** в корне репозитория — код **официального** [FaceFusion](https://github.com/facefusion/facefusion) (в сам PyPI-пакет `deepfake-benchmark` он не входит)

### Почему FaceFusion отдельно

Бенчмарк только вызывает `facefusion/facefusion.py` как subprocess. Репозиторий FaceFusion — чужой проект с собственными релизами и лицензией; его **не вшивают** в wheel пакета, иначе пришлось бы тащить гигабайты моделей/зависимостей и дублировать upstream.

Новому пользователю нужно **один раз** получить каталог `facefusion/` одним из способов:

1. **Скрипт из этого репозитория** (рекомендуется):

   ```bash
   poetry run python scripts/setup_facefusion.py
   ```

   Опционально зафиксировать версию (тег или ветка, если доступны при shallow clone):

   ```bash
   poetry run python scripts/setup_facefusion.py --ref 3.0.0
   ```

2. **Вручную**:

   ```bash
   git clone https://github.com/facefusion/facefusion.git facefusion
   ```

3. **Git submodule** (удобно для команды и фиксированного коммита upstream):

   ```bash
   git submodule add https://github.com/facefusion/facefusion.git facefusion
   git submodule update --init --recursive
   ```

Дальше — зависимости FaceFusion в **отдельном** conda/venv (см. раздел про GPU и `facefusion_python`), не обязательно в том же venv, что и Poetry.

### Зависимости FaceFusion (откуда ставить)

**Poetry ставит только бенчмарк** (`pyproject.toml`). Зависимости самого FaceFusion в это окружение **не входят** — их ставят в то же окружение, чей `python.exe` вы укажете в `facefusion_python`.

| Файл | Назначение |
|------|------------|
| `facefusion/requirements.txt` | Официальный список upstream (Gradio, ONNX и т.д.) — **рекомендуется** для полноценного FaceFusion |
| `deepfake_benchmark/core/generators/facefusion/requirements_facefusion.txt` | Урезанный набор пинов под headless (без UI), для воспроизводимости или если не нужен Gradio |

**Примеры после клонирования:**

```bash
# Вариант A — как у авторов FaceFusion (полный requirements.txt)
conda activate facefusion
python -m pip install -r facefusion/requirements.txt
```

```bash
# Вариант B — минимальный список из репозитория бенчмарка
conda activate facefusion
python -m pip install -r deepfake_benchmark/core/generators/facefusion/requirements_facefusion.txt
```

Одной командой с клоном и установкой в **уже выбранный** интерпретатор conda:

```bash
poetry run python scripts/setup_facefusion.py --install-deps --python C:/Users/ИМЯ/anaconda3/envs/facefusion/python.exe
```

По умолчанию `--install-deps` читает `facefusion/requirements.txt`. Чтобы поставить минимальный файл бенчмарка:

```bash
poetry run python scripts/setup_facefusion.py --install-deps --python C:/Users/ИМЯ/anaconda3/envs/facefusion/python.exe --requirements-file deepfake_benchmark/core/generators/facefusion/requirements_facefusion.txt
```

Если не хотите коммитить `facefusion/` в свой форк бенчмарка, добавьте в `.gitignore` строку `facefusion/` и храните только инструкцию выше.

## Установка

```bash
poetry install
```

Проверка окружения:

```bash
poetry run python --version
poetry run python -m deepfake_benchmark.cli --help
```

## Структура данных

Базовый рабочий вариант данных:

```text
deepfake_benchmark/data/
  facefusion/
    real/
      source/
        1.jpg
        2.jpg
      target/
        1.jpg
        2.jpg
```

Результат сохраняется в `generated_root`, обычно:

```text
deepfake_benchmark/data/facefusion/fake/
```

## Конфиги: base + overrides

Основной шаблон:

- `configs/base_facefusion.yaml`

Переопределения:

- `configs/facefusion_gpu.yaml`
- `configs/facefusion_all_vs_all_gpu.yaml`
- `configs/facefusion_external_sources_gpu.yaml`

Пример override:

```yaml
extends: "configs/base_facefusion.yaml"
pairing_mode: "all_vs_all"
```

`extends` поддерживается рекурсивно. Если будет цикл, CLI завершится с ошибкой.

## Запуск

### Рекомендуемый быстрый запуск (one_for_all)

```bash
poetry run python -m deepfake_benchmark.cli --config configs/facefusion_gpu.yaml --device cuda
```

### all_vs_all (тяжелый режим)

```bash
poetry run python -m deepfake_benchmark.cli --config configs/facefusion_all_vs_all_gpu.yaml --device cuda
```

### external_sources

```bash
poetry run python -m deepfake_benchmark.cli --config configs/facefusion_external_sources_gpu.yaml --device cuda
```

## Режимы паринга

### one_for_all

- берет первый валидный `source`
- прогоняет его по всем `target`
- самый быстрый и удобный для smoke/debug

### all_vs_all

- учитываются только элементы с `role=source` и `role=target` (как в `one_for_all`)
- строится полное декартово произведение: каждый source с каждым target; пары с одинаковым `media_path` пропускаются
- число запусков ~ `|sources| × |targets|` (раньше ошибочно смешивались роли и считались все файлы попарно)

### external_sources

- `source` берется из `external_source_dir`
- `target`:
  - из датасета (`use_loader: true`)
  - или из `custom_target_dir` (`use_loader: false`)

## Ключевые поля конфига

- `datasets`: список датасетов (например `["facefusion"]`)
- `generators`: список генераторов (сейчас практический путь - `facefusion`)
- `real_data_root`: корень real-данных
- `generated_root`: куда складывать fake
- `device`: `cpu`, `cuda` или `gpu` (`gpu` нормализуется в `cuda`)
- `facefusion_python`: путь к `python.exe` окружения FaceFusion (conda/venv); **не** путь к cuDNN — см. раздел «Настройка GPU»
- `pairing_mode`: `one_for_all | all_vs_all | external_sources`
- `use_loader`: поведение в `external_sources`
- `external_source_dir`: папка с внешними source
- `custom_target_dir`: папка с target при `use_loader: false`

## Smoke-test

```bash
poetry run python -m tests.smoke_test
```

## Настройка GPU (важно)

### Что нужно указывать в конфиге

**Путь к cuDNN в YAML вручную не задаётся.** cuDNN — это системные библиотеки (DLL на Windows). Их нужно либо положить в `PATH`, либо они уже лежат внутри conda-окружения FaceFusion (`…/envs/facefusion/Library/bin`).

**В конфиге указывается только `facefusion_python`** — путь к `python.exe` того окружения, где установлен FaceFusion и ONNX Runtime с GPU. Бенчмарк сам добавляет в `PATH` каталоги этого окружения (в т.ч. `Library/bin`), чтобы загрузилась `cudnn64_9.dll`.

Итого для пользователя:

| Ситуация | Что делать |
|----------|------------|
| FaceFusion в отдельном conda/venv | Заполнить `facefusion_python` путём к `python.exe` этого окружения |
| cuDNN и CUDA уже в системном `PATH` | Можно оставить `facefusion_python: null` и полагаться на `device: cuda` |

FaceFusion всегда запускается отдельным subprocess; без доступа к cuDNN ONNX падает в CPU — генерация идёт, но **без ускорения GPU**.

### Почему при «всё на GPU» генерация не стала заметно быстрее

Даже при рабочем CUDA:

- Каждая пара source/target — **отдельный процесс** FaceFusion: накладные расходы на старт процесса и загрузку моделей повторяются.
- Часть шагов (детекция лиц, препроцессинг, I/O) может оставаться на CPU или занимать сопоставимое время с небольшим инференсом на GPU.
- Небольшие картинки (например 256×256) и короткий прогон — GPU может выигрывать **секунды на пару**, а не «в разы», особенно если раньше вы уже не упирались в чистый CPU-inference.

**Как убедиться, что реально используется GPU**

1. Во время генерации в другом окне: `nvidia-smi` — должен расти `GPU-Util` и процесс `python`.
2. Автопроверка в репозитории (если есть conda `facefusion` и тестовые картинки):

   ```bash
   poetry run pytest tests/test_facefusion_gpu_env.py -v
   ```

   Тест проверяет, что в stderr FaceFusion **нет** сообщения ONNX про отсутствующую `cudnn64_9.dll` при `--execution-provider cuda` (значит CUDA-провайдер грузится, а не тихо падает в CPU из‑за cuDNN).

3. Ручной контроль: запустить один раз FaceFusion из того же `facefusion_python` и посмотреть stderr — не должно быть повторяющихся `TryGetProviderInfo_CUDA` / `cudnn64_9.dll` / `missing`.

### Шаги для пользователя

**1. Убедитесь, что FaceFusion установлен в отдельном окружении**

Рекомендуется использовать conda:

```bash
conda create -n facefusion python=3.10
conda activate facefusion
pip install -r facefusion/requirements.txt
```

**2. Найдите путь к Python этого окружения**

```bash
# conda (Windows)
conda activate facefusion
where python
# → C:\Users\<ИМЯ>\anaconda3\envs\facefusion\python.exe

# conda (Linux/Mac)
which python
# → /home/<ИМЯ>/anaconda3/envs/facefusion/bin/python
```

**3. Пропишите путь в конфиге**

```yaml
# configs/base_facefusion.yaml (или ваш override)
facefusion_python: "C:/Users/ИМЯ/anaconda3/envs/facefusion/python.exe"
```

Или передайте через override-конфиг `configs/facefusion_gpu.yaml`:

```yaml
extends: "configs/base_facefusion.yaml"
facefusion_python: "C:/Users/ИМЯ/anaconda3/envs/facefusion/python.exe"
```

**4. Проверьте, что CUDA видна в том окружении**

```bash
conda activate facefusion
python -c "import onnxruntime; print(onnxruntime.get_available_providers())"
# Должно содержать: CUDAExecutionProvider
```

**5. Запускайте как обычно**

```bash
poetry run python -m deepfake_benchmark.cli --config configs/facefusion_gpu.yaml --device cuda
```

### Если cuDNN установлена системно

Если `cudnn64_9.dll` уже есть в `C:\Windows\System32` или в `PATH` — поле `facefusion_python` можно оставить `null`, и всё заработает без него.

Проверить:

```powershell
where cudnn64_9.dll
```

---

## Частые проблемы

### `ModuleNotFoundError` при запуске

- запускать из корня репозитория
- использовать `poetry run ...`, а не системный `python`

### GPU не используется / ошибка `cudnn64_9.dll missing`

Причина: cuDNN не доступна для subprocess FaceFusion.
Решение: задать `facefusion_python` в конфиге (см. раздел выше).

### Кажется, что процесс "завис"

Чаще всего это `all_vs_all` — очень много пар и долгий прогон.
Для быстрой проверки используйте `one_for_all`.

---

## Что важно помнить

- для проверки пайплайна всегда используйте `one_for_all`
- `all_vs_all` включайте осознанно и только на небольшом наборе изображений
- все команды запуска делать через `poetry run ...`
- `facefusion_python` нужен всем, у кого FaceFusion в отдельном окружении