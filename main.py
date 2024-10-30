from matplotlib import pyplot as plt
import cv2
import numpy as np


def show_all_images(*images):
    titles = [
        "Исходник",
        "Границы (библ. Кэнни)",
        "Контур (библ. Кэнни)",

        "Размытое",
        "Границы (наш Кэнни)",
        "Контур (наш Кэнни)",

        "Границы (библ. Собель)",
        "Бинаризация (библ. Собель)",
        "Контур (библ. Собель)",

        "Границы (наш Собель)",
        "Бинаризация (наш Собель)",
        "Контур (наш Собель)",
    ]

    plt.figure(figsize=(20, 16))

    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(4, 3, i + 1)
        plt.imshow(img, cmap='gray' if 'Ч/Б' in title or 'Бинаризация' in title or 'Границы' in title else None)
        plt.title(title)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def sobel_filter(gray_image, return_theta=False):
    # Фильтр Собеля для нахождения границ
    kernel_G_x = np.array([[-1, -2, -1],
                           [0, 0, 0],
                           [1, 2, 1]])

    kernel_G_y = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]])

    G_x = cv2.filter2D(gray_image, cv2.CV_64F, kernel_G_x)
    G_y = cv2.filter2D(gray_image, cv2.CV_64F, kernel_G_y)
    G = np.hypot(G_x, G_y)
    G_normalized = np.round((G / G.max()) * 255).astype(int)
    image_G = cv2.convertScaleAbs(G_normalized)
    if return_theta:
        theta = np.arctan2(G_y, G_x)
        return image_G, theta
    else:
        return image_G


def non_max_suppression(G, theta):
    # Получаем размеры матрицы
    M, N = G.shape
    # Создаем результирующую матрицу
    Z = np.zeros((M, N), dtype=np.int32)
    # Переводи радиан в градусы
    # max -> 180, min -> -180
    angle = theta * 180.0 / np.pi
    # Поскольку выбор соседних пикселей, например, для -180+45 и 45 один и тот же, мы можем ограничиться только верхней полусферой,
    # прибавив ко всем отрицательным значениям + 180 градусов
    # max -> 180, min -> 0
    angle[angle < 0] += 180
    # Перебор всех пикселей кроме тех, у кого присутствуют не все соседи
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            # Инициализация значений соседних пикселей
            q = 255
            r = 255
            # Примерно горизонтальное направление
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                # Пиксель слева
                r = G[i - 1, j]
                # Пиксель справа
                q = G[i + 1, j]
            # Направление примерно 45 градусов
            elif 22.5 <= angle[i, j] < 67.5:
                # Пиксель слева снизу
                r = G[i - 1, j - 1]
                # Пиксель справа сверху
                q = G[i + 1, j + 1]
            # Направление примерно 90 градусов
            elif 67.5 <= angle[i, j] < 112.5:
                # Пиксель снизу
                r = G[i, j - 1]
                # Пиксель сверху
                q = G[i, j + 1]
            # Направление примерно 135 градусов
            elif 112.5 <= angle[i, j] < 157.5:
                # Пиксель слева сверху
                r = G[i - 1, j + 1]
                # Пиксель справа снизу
                q = G[i + 1, j - 1]
            # Сравниваем значений соседних пикселей с текущим
            if (G[i, j] >= q) and (G[i, j] >= r):
                Z[i, j] = G[i, j]
            else:
                Z[i, j] = 0
    # plt.imshow(G, cmap='gray')
    # plt.axis('off')
    # plt.show()
    # plt.imshow(Z, cmap='gray')
    # plt.axis('off')
    # plt.show()
    return Z


def threshold(img, low_threshold, high_threshold):
    img = np.round((img / img.max()) * 255).astype(int)
    # Получаем размеры матрицы
    M, N = img.shape
    # Создаем результирующую матрицу
    res = np.zeros((M, N), dtype=np.int32)
    # Значения неопределенных пикселей
    weak = 90
    # Значение пикселей, прошедших порог максимума
    strong = 255
    # Получение индексов пикселей, прошедших порог максимума
    strong_i, strong_j = np.where(img >= high_threshold)
    # # Получение индексов пикселей, прошедших порог минимума и непрошедших порог максимума
    weak_i, weak_j = np.where((img < high_threshold) & (img >= low_threshold))
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return res


def hysteresis(img):
    #Направления, в которых идем от сильного пикселя
    directions = np.array([[-1, -1, -1, 0, 0, 1, 1, 1,],
                           [-1, 0, 1, -1, 1, -1, 0, 1]])
    # Получаем размеры матрицы
    M, N = img.shape
    # Создаем результирующую матрицу
    res = np.zeros((M, N), dtype=np.int32)
    # Значение пикселей, прошедших порог максимума
    strong = 255
    # Перебор всех пикселей
    for i in range(1, M):
        for j in range(1, N):
            # Если сильный пиксель
            if img[i, j] == strong:
                # Записываем его в матрицу
                res[i, j] = strong
                # Идем от него в разные стороны
                for k in range(8):
                    dx = directions[0, k]
                    dy = directions[1, k]
                    x = i
                    y = j
                    while True:
                        # Делаем шаг в сторону
                        x += dx
                        y += dy
                        # Проверяем, не ушли ли за границу изображения
                        if x < 0 or y < 0 or x >= N or y >= M:
                            break
                        # Если встретили сильный или пустой пиксель
                        if img[x, y] == strong or img[x, y] == 0:
                            break
                        # Заменяем слабый пиксель на сильный
                        res[x, y] = strong
    return res


def canny(gray_image, low_threshold, high_threshold):
    G, theta = sobel_filter(gray_image, return_theta=True)
    res = non_max_suppression(G, theta)
    res = threshold(res, low_threshold, high_threshold)
    res = hysteresis(res)
    res_normalized = np.round((res / res.max()) * 255).astype(int)
    image_res = cv2.convertScaleAbs(res_normalized)
    return image_res


# Обработка изображения
def image_processing(path):
    # Загрузка цветного изображения
    image = cv2.imread(path, cv2.IMREAD_COLOR)

    # Конвертация из BGR в RGB
    original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    contour_image_s_lib = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    contour_image_s_our = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    contour_image_c_lib = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    contour_image_c_our = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Размытие изображения
    if path == "sunflower.jpg" or path == "main_road.jpg":
        blurred_image = cv2.GaussianBlur(original_image, (25, 25), 0)  # кэнни: 11 для шарика, 25 для знака и подсолнуха
    elif path == "balloon.jpg":
        blurred_image = cv2.GaussianBlur(original_image, (11, 11), 0)
    elif path == "dog.jpg":
        blurred_image = cv2.GaussianBlur(original_image, (3, 3), 0)
    else:
        blurred_image = cv2.GaussianBlur(original_image, (21, 21), 0)

    # Преобразование обратно в BGR и в оттенки серого
    bgr_image = cv2.cvtColor(blurred_image, cv2.COLOR_RGB2BGR)
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

    # Фильтр Собеля для нахождения границ
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_image_lib = cv2.magnitude(sobel_x, sobel_y)
    sobel_image_lib = np.round((sobel_image_lib / sobel_image_lib.max()) * 255).astype(int)
    sobel_image_lib = cv2.convertScaleAbs(sobel_image_lib)

    sobel_image_our = sobel_filter(gray_image)

    # Алгоритм Кэнни для нахождения границ
    canny_image_lib = cv2.Canny(gray_image, threshold1=10, threshold2=70)  # 50, 150
    canny_image_our = canny(gray_image, 10, 70)

    # Бинаризация
    max_output_value = 255  # 35 47 sunfl
    neighborhood_size = 35
    subtract_from_mean = 20
    binarized_image_sobel_lib = cv2.adaptiveThreshold(sobel_image_lib,
                                                      max_output_value,
                                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                      cv2.THRESH_BINARY_INV,
                                                      neighborhood_size,
                                                      subtract_from_mean)

    max_output_value = 255
    neighborhood_size = 35
    subtract_from_mean = 20
    binarized_image_sobel_our = cv2.adaptiveThreshold(sobel_image_our,
                                                      max_output_value,
                                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                      cv2.THRESH_BINARY_INV,
                                                      neighborhood_size,
                                                      subtract_from_mean)

    # Поиск контуров
    sobel_contour_lib, hierarchy11 = cv2.findContours(binarized_image_sobel_lib, cv2.RETR_EXTERNAL,
                                                      cv2.CHAIN_APPROX_SIMPLE)
    sobel_contour_our, hierarchy12 = cv2.findContours(binarized_image_sobel_our, cv2.RETR_EXTERNAL,
                                                      cv2.CHAIN_APPROX_SIMPLE)

    canny_contour_lib, hierarchy21 = cv2.findContours(canny_image_lib, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    canny_contour_our, hierarchy22 = cv2.findContours(canny_image_our, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Отрисовка контуров на исходнике
    cv2.drawContours(contour_image_s_lib, sobel_contour_lib, -1, (0, 255, 0), 7)
    cv2.drawContours(contour_image_s_our, sobel_contour_our, -1, (0, 255, 0), 7)

    cv2.drawContours(contour_image_c_lib, canny_contour_lib, -1, (0, 255, 0), 7)
    cv2.drawContours(contour_image_c_our, canny_contour_our, -1, (0, 255, 0), 7)

    show_all_images(original_image, canny_image_lib, contour_image_c_lib,
                    blurred_image, canny_image_our, contour_image_c_our,
                    sobel_image_lib, binarized_image_sobel_lib, contour_image_s_lib,
                    sobel_image_our, binarized_image_sobel_our, contour_image_s_our)


# Загрузка цветных изображений
path_seal = "seal.jpg"  # Тюленчик
path_sunflower = "sunflower.jpg"  # Подсолнух
path_sign = "main_road.jpg"  # Знак
path_balloon = "balloon.jpg"  # Шарик
path_dog = "dog.jpg"  # Бобака
path_chess = "chess.jpg"  # Пешка
path_arbuz = "5.jpg"  # Арбуз

# image_processing(path_arbuz)
# image_processing(path_chess)
# image_processing(path_dog)
# image_processing(path_seal)
image_processing(path_sunflower)
# image_processing(path_sign)
# image_processing(path_balloon)
