from matplotlib import pyplot as plt
import cv2
import numpy as np


def show_all_images(*images):
    titles = [
        "Исходник",
        "Размытое",
        "Ч/Б",
        "Границы (Собель)",
        "Границы (Кэнни)",
        "Бинаризация (Собель)",
        "Бинаризация (Кэнни)",
        "Контур (Собель)",
        "Контур (Кэнни)"
    ]

    plt.figure(figsize=(20, 16))

    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(3, 3, i + 1)
        plt.imshow(img, cmap='gray' if 'Ч/Б' in title or 'Бинаризация' in title else None)
        plt.title(title)
        plt.axis("off")

    plt.show()


# Обработка изображения
def image_processing(path):
    # Загрузка цветного изображения
    image = cv2.imread(path, cv2.IMREAD_COLOR)

    # Конвертация из BGR в RGB
    original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    contour_image_s = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    contour_image_c = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Размытие изображения
    blurred_image = cv2.GaussianBlur(original_image, (27, 27), 0)

    # Преобразование обратно в BGR и в оттенки серого
    bgr_image = cv2.cvtColor(blurred_image, cv2.COLOR_RGB2BGR)
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

    # Фильтр Собеля для нахождения границ
    grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)

    grad_combined_image = cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0)
    sobel_image = cv2.convertScaleAbs(grad_combined_image)

    # Алгоритм Кэнни для нахождения границ
    canny_image = cv2.Canny(blurred_image, threshold1=50, threshold2=150)

    # Бинаризация
    max_output_value = 255
    neighborhood_size = 47
    subtract_from_mean = 9
    binarized_image_sobel = cv2.adaptiveThreshold(sobel_image,
                                                  max_output_value,
                                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                  cv2.THRESH_BINARY_INV,
                                                  neighborhood_size,
                                                  subtract_from_mean)

    max_output_value = 255
    neighborhood_size = 47
    subtract_from_mean = 9
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
    cv2.drawContours(contour_image_c, contours_canny, -1, (0, 255, 0), 3)

    show_all_images(original_image, blurred_image, gray_image,
                    sobel_image, canny_image, binarized_image_sobel,
                    binarized_image_canny, contour_image_s, contour_image_c)


# Загрузка цветных изображений
path_seal = "seal.jpg"  # Тюленчик
path_sunflower = "sunflower.jpg"  # Подсолнух
path_sign = "main_road.jpg"  # Знак
path_balloon = "balloon.jpg"  # Шарик

image_processing(path_seal)
image_processing(path_sunflower)
image_processing(path_sign)
image_processing(path_balloon)
