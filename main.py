from matplotlib import pyplot as plt
import cv2
import numpy as np


# Вывод изображения
def show_image(image, title=None):
    if title == "Ч/Б" or title == "Бинаризация":
        plt.imshow(image, cmap="gray")
    else:
        plt.imshow(image)
    plt.axis("off")
    if title:
        plt.title(title)
    plt.show()


# Обработка изображения
def image_processing(image):
    # Конвертация из BGR в RGB
    original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Размытие изображения
    blurred_image = cv2.GaussianBlur(original_image, (27, 27), 0)

    # Преобразование обратно в BGR и в оттенки серого
    bgr_image = cv2.cvtColor(blurred_image, cv2.COLOR_RGB2BGR)
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

    # Фильтр Собеля для нахождения границ
    grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)

    grad_combined_image = cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0)
    grad_combined_image = cv2.convertScaleAbs(grad_combined_image)

    # Бинаризация
    max_output_value = 255
    neighborhood_size = 99
    subtract_from_mean = 10
    image_binarized = cv2.adaptiveThreshold(grad_combined_image,
                                            max_output_value,
                                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY,
                                            neighborhood_size,
                                            subtract_from_mean)

    # Инверсия бинарного изображения
    binarized_image = cv2.bitwise_not(image_binarized)

    # Поиск контуров
    contours, hierarchy = cv2.findContours(binarized_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Отрисовка контуров на исходнике
    contoured_image = original_image
    cv2.drawContours(original_image, contours, -1, (0, 255, 0), 3)

    # Вывод изображений
    show_image(original_image, title="Исходник")
    show_image(blurred_image, title="Размытое")
    show_image(gray_image, title="Ч/Б")
    show_image(grad_combined_image, title="Границы")
    show_image(binarized_image, title="Бинаризация")
    show_image(contoured_image, title="Контур")


# Загрузка цветных изображений
image_seal = cv2.imread("seal.jpg", cv2.IMREAD_COLOR)  # Тюленчик
image_sunflower = cv2.imread("sunflower.jpg", cv2.IMREAD_COLOR)  # Подсолнух
image_sign = cv2.imread("main_road.jpg", cv2.IMREAD_COLOR)  # Знак
image_balloon = cv2.imread("balloon.jpg", cv2.IMREAD_COLOR)  # Шарик

image_processing(image_seal)
# image_processing(image_sunflower)
# image_processing(image_sign)
# image_processing(image_balloon)
