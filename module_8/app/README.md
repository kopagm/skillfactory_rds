# Flask приложение для разворачивания модели

## Структура файлов

*server.py* - flask приложение. 

*post_image.sh* - bash скрипт для отправки POST запроса с файлом изображения.

*requirements.txt* - список зависимостей

## Использование

Для запуска необходимо разместить в текущей директории файл *model.hdf5* с готовой моделью.

При получении POST запроса с файлом картинки по адресу http://localhost:5000/predict возвражает json предсказание в виде `{"result": predict}`