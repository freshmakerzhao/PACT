import matplotlib.pyplot as plt


def init_render(image):
    ax = plt.subplot()
    plt_img = ax.imshow(image)
    plt.ion()
    return plt_img


def update_render(plt_img, image, pause_time=0.02):
    plt_img.set_data(image)
    plt.pause(pause_time)


def close_render():
    plt.close()

