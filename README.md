# Семантическая сегментация еды (FoodSeg103)

Проект по семантической сегментации блюд на датасете [FoodSeg103](https://github.com/LARC-CMU-SMU/FoodSeg103): 103 класса продуктов плюс фон. Разметка приходит в формате Supervisely (JPEG + JSON с zlib-сжатыми bitmap-масками).

Итоговая модель - **Mask2Former** (Swin-Small, предобучение на ADE20K). Более ранние эксперименты с CNN/transformer-декодерами из `segmentation_models_pytorch` лежат в `main.ipynb`.

## Данные

Ожидаемая раскладка (папка `dataset/` в `.gitignore`):

```
dataset/
  meta.json
  train/img/   train/ann/
  test/img/    test/ann/
```

В датасете **4983** train и **2135** test сэмплов. Редкие расхождения размера JPEG и поля `size` в JSON обрабатываются транспонированием или nearest-resize маски.

Утилита `size print.py` сканирует картинки и пишет гистограмму уникальных разрешений.

## Структура репозитория

| Файл                | Назначение                                         |
| ------------------- | -------------------------------------------------- |
| `mask2former.ipynb` | Обучение, оценка и визуализация **лучшей** модели  |
| `main.ipynb`        | Чтение датасета и эксперименты с SMP-архитектурами |
| `size print.py`     | Статистика размеров изображений                    |

## Какие модели пробовали

### 1. Семейство SMP (`main.ipynb`)

Единый пайплайн: ImageNet-энкодер, логиты без активации, **Dice Loss** (`multiclass`), **Adam** `1e-4`, 10 эпох, batch 5. Картинки масштабируются по длинной стороне и паддятся до **576×576**. Аугментации: горизонтальный/вертикальный флип, поворот на 90°, яркость.

Декодеры, которые переключались:

- **DeepLabV3Plus**
- **UPerNet**
- **Segformer**

Энкодеры, которые перебирались (`BASE_ENCODER`):

- `resnet18`, `resnet34`, `resnet50`, `mobilenet_v2`, `senet154`.

Веса сохранялись как `{encoder}_food_seg.pth`; лучший чекпоинт выбирался по val IoU.

Метрики для **UPerNet + ResNet50** (лучшие метрики из всех проверенных комбинаций декодеров и архитектур)

| Pixel acc | Dice loss | mean IoU |
| --------- | --------- | -------- |
| 0.8386    | 0.0733    | 0.0940   |

Высокая pixel accuracy при низком mIoU для этого датасета объясняется тем, что фон и крупные блюда доминируют, редкие классы почти не детектятся. Это и стало причиной перехода к Mask2Former.

### 2. Mask2Former (`mask2former.ipynb`) - лучшая модель

Query-based universal segmentation: Swin backbone + pixel decoder + masked-attention transformer decoder.

Голова классификации переинициализируется (`ignore_mismatched_sizes`): ADE20K 150 классов → **104** метки FoodSeg (103 + background). Класс-предиктор в HF имеет размер `num_labels + 1` (служебный no-object).

Результат: **mean IoU 0.1757**, pixel acc 0.2715, loss ~2.0. Сравнивать mIoU с `main.ipynb` нужно по `val_miou_resized` на разрешении модели; полный проход с `evaluate.mean_iou` и `ignore_index=0` включается флагом `VAL_COMPUTE_MIOU`.

---

## Mask2Former: обучение и оптимизации

### Препроцессинг и аугментации

Сырые RGB и маски грузятся без ресайза (`FoodSegRawDataset`), затем Albumentations:

- train: HFlip, VFlip, RandomRotate90, ShiftScaleRotate, яркость/контраст, HSV, GaussNoise **или** GaussianBlur;
- и train, и val: `LongestMaxSize` + pad до `IMAGE_SIZE = 864`, ImageNet-нормализация.

`Mask2FormerImageProcessor` только собирает `pixel_values` / `mask_labels` / `class_labels` (`do_resize/do_rescale/do_normalize=False`). Фон в лоссе: `ignore_index=0`.

Оригинальные картинки и маски сохраняются в батче для оценки на нативном разрешении.

### Как ужимается VRAM

Mask2Former на 864px легко упирается в память. В ноутбуке собраны приёмы, без которых обучение на одной GPU не сходилось:

1. **Point sampling лосса.** `config.train_num_points = 4096` вместо дефолтных 12544.
2. **Пропуск auxiliary loss без поломки forward** (`SKIP_AUXILIARY_LOSS`).
3. **AMP +** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
4. **Накопление градиента** вместо большого batch.
5. **Периодический** `torch.cuda.empty_cache()` каждые `EMPTY_CACHE_EVERY=25` батчей.
6. **Val mIoU не на каждом батче каждой эпохи.**

### Инференс

`processor.post_process_semantic_segmentation` с `target_sizes` исходного или 864×864 кадра. Визуализация как в `main.ipynb`: GT, предсказание, совпадение цветом класса / ошибка оранжевой штриховкой.

## Как запустить

Нужны CUDA, датасет в `dataset/` (или `HSS Project/dataset/`, ноутбуки это подхватывают).

```text
pip install torch torchvision transformers albumentations opencv-python evaluate tqdm matplotlib pandas Pillow segmentation-models-pytorch
```

- SMP-бейзлайн: `main.ipynb`
- Mask2Former: ячейки `mask2former.ipynb` сверху вниз

## Зависимости по стеку

- **SMP-эксперименты:** PyTorch, `segmentation_models_pytorch`, torchvision.
- **Mask2Former:** `transformers`, `albumentations`, `evaluate` (метрика `mean_iou` с Hub).
