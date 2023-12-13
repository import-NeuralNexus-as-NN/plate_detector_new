import os
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


def detecting_plate(image_path, new_path):
    # Загрузка изображения
    image = cv2.imread(image_path)

    # Загрузка предварительно обученного детектора
    cascade_path = 'haarcascade_russian_plate_number.xml'
    plate_cascade = cv2.CascadeClassifier(cascade_path)

    # Преобразование в оттенки серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Детекция номера
    plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=7, minSize=(30, 30))

    # Находим самый большой номер
    largest_plate = None
    largest_area = 0

    for (x, y, w, h) in plates:
        # Вычисляем площадь
        area = w * h

        # Если текущий номер больше предыдущего самого большого
        if area > largest_area:
            largest_area = area
            largest_plate = (x, y, w, h)

    # Если найден номер, обрезаем изображение
    if largest_plate is not None:
        x, y, w, h = largest_plate
        cropped_plate = image[y:y + h, x:x + w]

        # Преобразование в оттенки серого
        gray_cropped = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)

        # Применение адаптивной бинаризации
        _, thresh = cv2.threshold(gray_cropped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Применение морфологических операций
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # Детекция контуров
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Находим контур с максимальной длиной
        largest_contour = max(contours, key=cv2.contourArea)

        # Находим прямоугольник, описывающий контур
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # Поворачиваем и наклоняем изображение для выравнивания номера
        width, height = int(rect[1][0]), int(rect[1][1])
        src_pts = box.astype("float32")
        dst_pts = np.array([[0, height - 1], [0, 0], [width - 1, 0], [width - 1, height - 1]], dtype="float32")

        # Получаем матрицу преобразования и применяем ее
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        corrected_plate = cv2.warpPerspective(cropped_plate, M, (width, height))
        if width < height:
            corrected_plate = cv2.rotate(corrected_plate, cv2.ROTATE_90_CLOCKWISE)
        if width != 250 and height != 65:
            corrected_plate = cv2.resize(corrected_plate, (250, 65))
        # Выводим изображение с выровненным номером
        # cv2_imshow(corrected_plate)
        cv2.imwrite(new_path, corrected_plate)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    else:
        print('Номер не обнаружен.')


def split_plate(image_path):
    # Загрузка изображения с выровненным номером
    corrected_plate = cv2.imread(image_path)

    # Получаем размеры изображения
    height, width, _ = corrected_plate.shape

    # Процентное соотношение разделения
    percentage_split = 75.5
    percentage_bottom_cut = 42

    # Вычисляем границы разделения
    split_point = int(width * (percentage_split / 100))

    # Вырезаем и выводим две части
    first_part = corrected_plate[:, :split_point]
    second_part = corrected_plate[:, split_point:]

    # Вычисляем высоту второй части
    height_second_part = second_part.shape[0]

    # Вычисляем количество пикселей, которые нужно обрезать снизу
    cut_pixels = int(height_second_part * (percentage_bottom_cut / 100))

    # Обрезаем 20% снизу от второй части
    second_part_cropped = second_part[:-cut_pixels, :]

    return first_part, second_part_cropped


def extract_characters(image):
    # Преобразование в оттенки серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Применение адаптивной бинаризации для выделения символов
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Нахождение контуров символов
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Сортировка контуров по X-координате
    contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])

    # Итерация по контурам и выделение каждого символа с белым фоном
    characters = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Исключение слишком маленьких символов (можно настроить)
        if w > 10 and h > 10:
            # Вырезаем символ с добавлением белого фона
            symbol_with_background = 255 * np.ones((h + 10, w + 10, 3), dtype=np.uint8)
            symbol_with_background[5:5 + h, 5:5 + w, :] = image[y:y + h, x:x + w, :]

            characters.append(symbol_with_background)

    return characters


def split_symbols(plate_path, first_path, second_path):
    first_part, second_part_cropped = split_plate(plate_path)
    # Выделение символов для первой и второй частей
    characters_first_part = extract_characters(first_part)
    characters_second_part = extract_characters(second_part_cropped)

    save_path_first_part = first_path
    save_path_second_part = second_path

    # Создание папок, если они не существуют
    os.makedirs(save_path_first_part, exist_ok=True)
    os.makedirs(save_path_second_part, exist_ok=True)

    # Сохранение изображений символов
    for i, character in enumerate(characters_first_part, start=1):
        save_filename = os.path.join(save_path_first_part, f"character_{i}.png")
        cv2.imwrite(save_filename, character)

    for i, character in enumerate(characters_second_part, start=1):
        save_filename = os.path.join(save_path_second_part, f"character_{i}.png")
        cv2.imwrite(save_filename, character)

    cv2.destroyAllWindows()


class_to_idx = {'0': 0,
                '1': 1,
                '2': 2,
                '3': 3,
                '4': 4,
                '5': 5,
                '6': 6,
                '7': 7,
                '8': 8,
                '9': 9,
                'A': 10,
                'B': 11,
                'C': 12,
                'E': 13,
                'H': 14,
                'K': 15,
                'M': 16,
                'P': 17,
                'T': 18,
                'X': 19,
                'Y': 20,
                'negative': 21}
idx_to_class = {i: class_name for class_name, i in class_to_idx.items()}


def predict_image(image_path, loaded_model):
    # Загрузка изображения и применение трансформаций
    image = Image.open(image_path).convert('L')  # 'L' означает оттенки серого
    transform = transforms.Compose([
        transforms.Resize((55, 55)),
        transforms.ToTensor(),
    ])
    image = transform(image).unsqueeze(0)  # Добавляем размерность батча

    # Переводим модель в режим оценки (не обучения)
    loaded_model.eval()

    # Передаем изображение в модель для предсказания
    with torch.no_grad():
        output = loaded_model(image)

    # Получаем предсказанный класс
    _, predicted_class = torch.max(output, 1)

    # Возвращаем название класса
    predicted_class_name = idx_to_class[predicted_class.item()]
    if predicted_class_name == 'negative':
        predicted_class_name = ''

    return predicted_class_name


# функция удаления файлов в папке
def delete_files_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f'Ошибка при удалении файла {file_path}. {e}')


def create_predictions(folder_path, loaded_model):
    prediction = []
    for filename in sorted(os.listdir(folder_path)):
        image_path = os.path.join(folder_path, filename)
        symbol = predict_image(image_path, loaded_model)
        prediction.append(symbol)

    if not prediction:
        prediction = ['Извините, номер не найден']
    return prediction



class CustomCNN(nn.Module):
    def __init__(self, num_classes=22):
        super(CustomCNN, self).__init__()

        # Сверточные слои
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        # Слои объединения (pooling)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Полносвязанные слои
        self.fc1 = nn.Linear(128 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Проход через сверточные слои
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # Разворачиваем тензор перед подачей на полносвязанные слои
        x = x.view(-1, 128 * 6 * 6)

        # Проход через полносвязанные слои
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x