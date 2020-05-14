import matplotlib.pyplot as plt
import numpy as np


def full(resolution):
    i = 0
    indexes_array = np.zeros((resolution * resolution, 2))
    for j in range(resolution):
        for k in range(resolution):
            indexes_array[i] = np.array([j, k])
            i += 1
    indexes_array = indexes_array - (0.5 * resolution)
    return np.array(indexes_array)


def random_cols(resolution, num_of_cols_to_sample):
    cols = np.random.randint(0, resolution, int(num_of_cols_to_sample))
    indexes_array = []
    for col in cols:
        col_indexes = [[i, col] for i in range(resolution)]
        indexes_array = indexes_array + col_indexes
    return np.array(indexes_array)


def random_rows(resolution, num_of_rows_to_sample):
    rows = np.random.randint(0, resolution, int(num_of_rows_to_sample))
    indexes_array = []
    for row in rows:
        col_indexes = [[row, i] for i in range(resolution)]
        indexes_array = indexes_array + col_indexes
    return np.array(indexes_array)


def circle(resolution, number_of_samples):
    # the center of the circle will be in the middle of the image
    center = (resolution + 1) / 2
    # the radius of the allowed circle can be calculated:
    # the volume of the circle is the amount of samples needed
    # pi*r^2 = #n
    # r^2 = #n / pi
    # r = sqrt(#n / pi)
    # we add epsilon because we prefer to take many samples and cut short then take too little and have no way to fix it
    radius = np.sqrt(number_of_samples / np.pi) + 1e-3
    # we will count how many pixels we have created, so we will not create more then allowed
    couner = 0
    indexes = []
    for i in range(resolution):
        for j in range(resolution):
            # check if the distance from the center of the circle is in the allowed radius
            if np.linalg.norm(np.array([i - center, j - center]), 2) <= radius:
                indexes += [[i, j]]
                couner += 1
                if couner == number_of_samples:
                    return np.array(indexes)
    return np.array(indexes)


def spiral(resolution, samples, density):
    # the spiral needs to follow the cartesian equation formula
    # r = a*theta
    indexes_array = []
    middle = (resolution + 1) / 2
    for i in range(samples):
        t = i / (samples / 10) * np.pi
        x = (1 + 5 * t) * np.cos(density * t)
        y = (1 + 5 * t) * np.sin(density * t)
        indexes_array += [[x - middle, y - middle]]
    return np.round(np.array(indexes_array)).astype(int)


def random_dots(resolution, number_of_samples):
    return np.random.randint(0, resolution - 1, (number_of_samples, 2))


def to_trajectory_image(resolution, indexes_array):
    indexes_array = np.round(indexes_array).astype(int)
    image = np.zeros((resolution, resolution))
    image[indexes_array[:, 0], indexes_array[:, 1]] = 1
    return image


def plot_trajectory(resolution, indexes_array, title):
    plt.figure()
    plt.imshow(to_trajectory_image(resolution, indexes_array), cmap='gray')
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    plot_trajectory(320, spiral(320, 4266, 5), 'WOW!')
