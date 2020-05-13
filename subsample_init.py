import matplotlib.pyplot as plt
import numpy as np
import torch


class TrajectoryInit:
    def __init__(self, resolution, device):
        self.resolution = resolution
        self.device = device
        self.trajectory = self.full()

    def full(self):
        index = 0
        every_point = torch.zeros(self.resolution * self.resolution, 2)
        for i in range(self.resolution):
            # if 140 <= i <= 170:
            # continue
            for j in range(self.resolution):
                # if 120 <= j <= 160:
                # continue
                every_point[index] = torch.tensor([i, j])
                index += 1
        every_point = every_point - (0.5 * self.resolution)
        every_point = every_point.to(self.device)
        return every_point

    def subsample_random_cols(self, num_of_cols_to_sample):
        cols = np.random.randint(0, self.resolution, num_of_cols_to_sample)
        indexes_array = []
        for col in cols:
            col_indexes = [[i, col] for i in range(self.resolution)]
            indexes_array = indexes_array + col_indexes
        return torch.tensor(np.array(indexes_array))

    def subsample_random_rows(self, num_of_rows_to_sample):
        rows = np.random.randint(0, self.resolution, num_of_rows_to_sample)
        indexes_array = []
        for row in rows:
            col_indexes = [[row, i] for i in range(self.resolution)]
            indexes_array = indexes_array + col_indexes
        return torch.tensor(np.array(indexes_array))

    def spiral(self, samples, density):
        # the spiral needs to follow the cartesian equation formula
        # r = a*theta
        indexes = []
        middle = (self.resolution + 1) / 2
        for i in range(samples):
            t = i / (samples / 10) * np.pi
            x = (1 + 5 * t) * np.cos(density * t)
            y = (1 + 5 * t) * np.sin(density * t)
            indexes += [[x - middle, y - middle]]
        return torch.tensor(np.array(indexes))

    def random_dots(self, number_of_samples):
        return torch.tensor(np.random.randint(0, self.resolution - 1, (number_of_samples, 2)))

    def plot_trajectory(self, indexes_array, title):
        image = np.zeros((self.resolution, self.resolution))
        image[indexes_array[:, 0], indexes_array[:, 1]] = 1
        plt.figure()
        plt.imshow(image, cmap='gray')
        plt.title(title)
        plt.show()
