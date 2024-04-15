from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import logging
import joblib
from load import load_dataset

number = 7
MODEL_NAME = "model_" + str(number) + ".joblib"
LOG_FILENAME = "model_" + str(number) + ".log"
N_ESTIMATORS = 500  # Дефолтное 500
MAX_DEPTH = 20  # Дефолтное 20
MIN_SAMPLES_SPLIT = 30  # Дефолтное 10

# Настройка логирования
logging.basicConfig(filename=LOG_FILENAME, level=logging.INFO, format='%(asctime)s - %(message)s')

# Функция для загрузки датасета
# возвращает numpy массивы: данные и метки классов
pictures, labels = load_dataset("D:\dataset")

# Разбиваем датасет на тренировочные, валидационные и демонстрационные данные
pic_train, pic_temp, label_train, label_temp = train_test_split(pictures, labels, test_size=0.4, stratify=labels,
                                                                random_state=42)
pic_val, pic_demo, label_val, label_demo = train_test_split(pic_temp, label_temp, test_size=0.5, stratify=label_temp,
                                                            random_state=42)

print(label_train)

# Инициализируем модель
model = RandomForestClassifier(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, min_samples_split=MIN_SAMPLES_SPLIT,
                               random_state=42)

# Преобразование трехмерных изображений в одномерные векторы
pic_train_flattened = pic_train.reshape(pic_train.shape[0], -1)
pic_val_flattened = pic_val.reshape(pic_val.shape[0], -1)
pic_demo_flattened = pic_demo.reshape(pic_demo.shape[0], -1)
pic_flattened = pictures.reshape(pictures.shape[0], -1)

# Обучения модели
model.fit(pic_train_flattened, label_train)

# На валидационных данных
label_pred_val = model.predict(pic_val_flattened)
accuracy_val = accuracy_score(label_val, label_pred_val)
f1_val = f1_score(label_val, label_pred_val, average='weighted')  # Using weighted average for multi-class scenario
logging.info(f'Точность на валидационных данных: {accuracy_val}')
logging.info(f'F1 на валидационных данных:: {f1_val}')
print(f'Точность на валидационных данных: {accuracy_val}')
print(f'F1 на валидационных данных:: {f1_val}')
report_val = classification_report(label_val, label_pred_val)
logging.info('Classification на валидационных данных::\n' + report_val)
print('Classification на валидационных данных:\n', report_val)

# Тестирование на демонстрационных данных
label_pred_demo = model.predict(pic_demo_flattened)
f1_demo = f1_score(label_demo, label_pred_demo, average='weighted')
accuracy_demo = accuracy_score(label_demo, label_pred_demo)
logging.info(f'Точность на демонстрационных данных: {accuracy_demo}')
print(f'Точность на демонстрационных данных: {accuracy_demo}')
logging.info(f'F1 на демонстрационных данных: {f1_demo}')
print(f'F1 на демонстрационных данных: {f1_demo}')
report_demo = classification_report(label_demo, label_pred_demo)
logging.info('Classification на демонстрационных данных:\n' + report_demo)
print('Classification на демонстрационных данных:\n', report_demo)


# Тестирование на всех данных
label_pred_all = model.predict(pic_flattened)
f1_all = f1_score(labels, label_pred_all, average='weighted')
print(label_pred_all)
print(labels)
accuracy_demo = accuracy_score(labels, label_pred_all)
logging.info(f'Точность на всех данных: {accuracy_demo}')
print(f'Точность на всех данных: {accuracy_demo}')
logging.info(f'F1 score на всех данных: {f1_all}')
print(f'F1 на всех данных: {f1_all}')
report_all = classification_report(labels, label_pred_all)
logging.info('Classification report на всех данных:\n' + report_all)
print('Classification report на всех данных:\n', report_all)

# Сохраняем модель
joblib.dump(model, MODEL_NAME)
