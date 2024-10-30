from matplotlib import pyplot as plt
import cv2
import numpy as np


def show_all_images(*images):
    titles = [
        "Исходник",
        "Размытое",
        "Ч/Б",
        "Границы (Собель)",
        "Бинаризация (Собель)",
        "Контур (Собель)",
        "Границы (Кэнни)",
        "Бинаризация (Кэнни)",
        "Контур (Кэнни)"
    ]

    plt.figure(figsize=(20, 16))

    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(3, 3, i + 1)
        plt.imshow(img, cmap='gray' if 'Ч/Б' in title or 'Бинаризация' in title or 'Границы' in title else None)
        plt.title(title)
        plt.axis("off")

    plt.show()

def sobel_filter(gray_image):
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
    return image_G

# Обработка изображения
def image_processing(path):
    # Загрузка цветного изображения
    image = cv2.imread(path, cv2.IMREAD_COLOR)

    # Конвертация из BGR в RGB
    original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    contour_image_s = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    contour_image_c = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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
    sobel_image = sobel_filter(gray_image)

    # Алгоритм Кэнни для нахождения границ
    canny_image = cv2.Canny(blurred_image, threshold1=50, threshold2=150)  # 50, 150

    # Бинаризация
    max_output_value = 255  # 35 47 sunfl
    neighborhood_size = 35
    subtract_from_mean = 47
    binarized_image_sobel = cv2.adaptiveThreshold(sobel_image,
                                                  max_output_value,
                                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                  cv2.THRESH_BINARY_INV,
                                                  neighborhood_size,
                                                  subtract_from_mean)

    max_output_value = 255
    neighborhood_size = 47
    subtract_from_mean = 5
    binarized_image_canny = cv2.adaptiveThreshold(canny_image,
                                                  max_output_value,
                                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                  cv2.THRESH_BINARY_INV,
                                                  neighborhood_size,
                                                  subtract_from_mean)

    # Поиск контуров
    contours_sobel, hierarchy1 = cv2.findContours(binarized_image_sobel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_canny, hierarchy2 = cv2.findContours(binarized_image_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Отрисовка контуров на исходнике
    cv2.drawContours(contour_image_s, contours_sobel, -1, (0, 255, 0), 3)
    cv2.drawContours(contour_image_c, contours_canny, -1, (0, 255, 0), 7)

    show_all_images(original_image, blurred_image, gray_image,
                    sobel_image, binarized_image_sobel, contour_image_s,
                    canny_image, binarized_image_canny, contour_image_c)


# Загрузка цветных изображений
path_seal = "seal.jpg"  # Тюленчик
path_sunflower = "sunflower.jpg"  # Подсолнух
path_sign = "main_road.jpg"  # Знак
path_balloon = "balloon.jpg"  # Шарик
path_dog = "dog.jpg"  # Бобака
path_chess = "chess.jpg" # Пешка



# image_processing("5.jpg")
# image_processing(path_chess)
# image_processing(path_dog)
# image_processing(path_seal)
# image_processing(path_sunflower)
# image_processing(path_sign)
# image_processing(path_balloon)
