import numpy as np
import os
import matplotlib.image as mpimg
from skimage.transform import resize


def load_dataset(base_path, img_size=(400, 400)):
    """
    Загружает датасет изображений, преобразуя их к единому размеру с использованием matplotlib и skimage.

    :param base_path: Путь к каталогу с подкаталогами 'cat' и 'dog'.
    :param img_size: Желаемый размер изображений после ресайза.
    :return: кортеж из двух numpy массивов: массив данных и массив меток.
    """
    categories = ['cat', 'dog']  # Названия подкаталогов для классов
    data = []  # Для хранения данных изображений
    labels = []  # Для хранения меток классов (0 для кошек, 1 для собак)

    for label, category in enumerate(categories):
        cat_dir = os.path.join(base_path, category)
        for img_name in os.listdir(cat_dir):
            img_path = os.path.join(cat_dir, img_name)
            try:
                img = mpimg.imread(img_path)  # Чтение изображения
                img = resize(img, img_size, anti_aliasing=True,
                             mode='reflect')  # Изменение размера с сохранением пропорций

                data.append(img)
                labels.append(label)
            except Exception as e:
                print(f'Ошибка при обработке изображения {img_path}: {e}')

    # Преобразование списков в numpy массивы
    data = np.array(data, dtype='float32') / 255.0  # Нормализация данных
    labels = np.array(labels)

    return data, labels
