from matplotlib import pyplot as plt
import cv2
import numpy as np


def show_image(image, title=None):
    if title == "Ч/Б":
        plt.imshow(image, cmap="gray")
    else:
        plt.imshow(image)
    plt.axis("off")
    if title:
        plt.title(title)
    plt.show()


# Загрузка цветных изображений
image_seal = cv2.imread("seal.jpg", cv2.IMREAD_COLOR)
image_sunflower = cv2.imread("sunflower.jpg", cv2.IMREAD_COLOR)

# Конвертация из BGR в RGB
original_seal = cv2.cvtColor(image_seal, cv2.COLOR_BGR2RGB)
original_sunflower = cv2.cvtColor(image_sunflower, cv2.COLOR_BGR2RGB)

# Размытие изображения
blurred_seal = cv2.GaussianBlur(original_seal, (11, 11), 0)
blurred_sunflower = cv2.GaussianBlur(original_sunflower, (11, 11), 0)

# Преобразование обратно в BRG и в оттенки серого
bgr_seal = cv2.cvtColor(blurred_seal, cv2.COLOR_RGB2BGR)
bgr_sunflower = cv2.cvtColor(blurred_sunflower, cv2.COLOR_RGB2BGR)
gray_seal = cv2.cvtColor(bgr_seal, cv2.COLOR_BGR2GRAY)
gray_sunflower = cv2.cvtColor(bgr_sunflower, cv2.COLOR_BGR2GRAY)

# Тюленчик
show_image(original_seal, title="Исходник")
show_image(blurred_seal, title="Размытое")
show_image(gray_seal, title="Ч/Б")

# Подсолнух
show_image(original_sunflower, title="Исходник")
show_image(blurred_sunflower, title="Размытое")
show_image(gray_sunflower, title="Ч/Б")
