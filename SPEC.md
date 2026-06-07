# SPEC.md

## 1. Обзор технической реализации

Проект реализован как Python-пакет с `pyproject.toml`, управлением зависимостями через `uv`, генераторами синтетических данных, модулями методов, метриками, визуализацией и CLI-раннерами экспериментов.

## 2. Структура проекта

- `src/dataset/`: генерация траекторий, сегментов и point cloud данных.
- `src/methods/`: адаптированные методы кластеризации, классификации, прогнозирования и трекинга.
- `src/metrics/`: расчет кластеризационных, классификационных, forecasting, tracking и resource метрик.
- `src/visualization/`: генерация figure-артефактов для результатов и отчета.
- `src/experiments/`: CLI-раннеры и сборка сводных артефактов.
- `data/`, `results/`, `report_assets/`, `report_template/`: данные, результаты и материалы для DOCX-пайплайна.

## 3. API и интерфейсы

- CLI:
  - `python -m src.dataset.generate_trajectories`
  - `python -m src.dataset.generate_segments`
  - `python -m src.dataset.generate_pointcloud`
  - `python -m src.experiments.run_all --dataset synthetic_v1 --output results`
- Report-prep CLI:
  - `python -m src.experiments.prepare_report_assets`
  - `python -m src.experiments.validate_outputs`
- Дополнительные CLI: `run_clustering`, `run_classification`, `run_forecasting`, `run_pointcloud`, `run_resnet_hdbscan`.

## 4. Данные и модели

- Траектории сохраняются в CSV с кинематическими признаками и разметкой паттернов.
- Сегменты сохраняются в NPZ со сплитами `train/val/test`.
- Point cloud данные сохраняются в CSV по уровням сложности `clean`, `medium`, `hard`.
- Нейросетевые модели включают 1D CNN классификатор, LSTM-прогноз и ResNet-like encoder.
- Raw результаты в `results/metrics/` сохраняются как machine-readable слой.
- Report-ready таблицы и markdown-артефакты формируются отдельно в `report_assets/tables/`, `results/tables/` и `report_assets/text_fragments/`.

## 5. Интеграции

- `uv` для управления зависимостями и окружением проекта.
- Базовый стек зависимостей включает `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `pyyaml`, `psutil`, `tqdm`, `pillow`, `torch`, `torchvision`, `hdbscan`, `umap-learn`, `nvidia-ml-py`.

## 6. Переменные окружения

- Обязательные переменные окружения не требуются.
- GPU используется автоматически при доступности через `torch.cuda`.

## 7. Запуск и деплой

- Установка и синхронизация зависимостей выполняется через `uv`.
- Основной воспроизводимый прогон выполняется через `uv run python -m src.experiments.run_all --dataset synthetic_v1 --output results`.
- Подготовка материалов для отчета выполняется через `python -m src.experiments.prepare_report_assets`.
- Проверка готовности выполняется через `python -m src.experiments.validate_outputs`, результат сохраняется в `VALIDATION_REPORT.md`.

## 8. Нерешенные технические вопросы

- При необходимости оптимизировать объем point cloud quick-profile и расширить итоговую сводку для полноразмерных прогонов.
- При переносе репозитория между машинами учитывать состояние Git LFS-артефактов в `data/pointcloud/`.
