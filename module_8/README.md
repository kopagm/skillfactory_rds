# Классификация модели автомобиля по фотографии
## Цель и задачи проекта.

Цель проекта - классифицировать автомобили по фотографии
с использованием Transfer Learning и Fine-tuning.

Соревнование kaggle - https://www.kaggle.com/c/sf-dl-car-classification


## Этапы выполнения

- Получение, распаковка данных из датасета соревнования
- EDA
- Аугментация и генерация изображений с использованием `tf.keras.preprocessing.image.ImageDataGenerator`
- Использование Transfer Learning и Fine-tuning.
- Использование EfficientNetB6, Xception с набором весов imagenet
- Использование Batch Normalization в архитектуре "головы"
- Выбор коэффициента скорости обучения
- Использование Test Time Augmentation
- Усреднение полученных предсказаний 
- Выводы