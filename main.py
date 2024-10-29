from matplotlib import pyplot as plt
import cv2
import numpy as np


# Вывод изображения
def show_image(image, title=None):
    if title == "Ч/Б":
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
    original_image = cv2.cvtColor(image_seal, cv2.COLOR_BGR2RGB)

    # Размытие изображения
    blurred_image = cv2.GaussianBlur(original_image, (11, 11), 0)

    # Преобразование обратно в BGR и в оттенки серого
    bgr_image = cv2.cvtColor(blurred_image, cv2.COLOR_RGB2BGR)
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

    # Вывод изображений
    show_image(original_image, title="Исходник")
    show_image(blurred_image, title="Размытое")
    show_image(gray_image, title="Ч/Б")


# Загрузка цветных изображений
image_seal = cv2.imread("seal.jpg", cv2.IMREAD_COLOR)  # Тюленчик
image_sunflower = cv2.imread("sunflower.jpg", cv2.IMREAD_COLOR)  # Подсолнух

image_processing(image_seal)
image_processing(image_sunflower)
