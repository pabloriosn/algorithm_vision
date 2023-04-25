import numpy as np
import cv2


class ParticlesFilter:
    def __init__(self, num_particles: int, particle_size: int, image_size: (int, int), resample: int = 10):

        self._num_particles: int = num_particles
        self._particle_size: int = particle_size
        self._w, self._h = image_size

        self._resample: int = resample

        self._weights = None
        self._particles = None

    def get_particles(self) -> np.ndarray:
        return self._particles

    def initialization(self):
        print("Initialization particles")

        # Initialize the particles
        self._particles = np.random.randint(low=0,
                                            high=[self._w - self._particle_size, self._h - self._particle_size],
                                            size=(self._num_particles, 2))

        # Initialize the weights with equal probability
        self._weights = np.ones(self._num_particles) / self._num_particles

    def evaluation(self, mask: np.ndarray):
        total_pixel = np.count_nonzero(mask)

        check = 0

        while check == 0:
            for i, pos in enumerate(self._particles):
                particle = mask[pos[0]:pos[0] + self._particle_size, pos[1]:pos[1] + self._particle_size]

                self._weights[i] = np.count_nonzero(particle) / (total_pixel + 1e-10)

            check = np.count_nonzero(self._weights)
            if check == 0:
                self.initialization()

        # Normalize the weights
        self._weights /= np.sum(self._weights)

    def estimation(self) -> (int, int):
        # Choose the particle with the highest weight
        max_weight = np.argmax(self._weights)

        return np.array([self._particles[max_weight]])

    def selection(self):
        cum_weights = np.cumsum(self._weights)

        for j in range(self._num_particles):
            pos = np.where(cum_weights >= np.random.uniform(0, 1))[0][0]

            self._particles[j] = self._particles[pos]

        self._weights = np.ones(self._num_particles) / self._num_particles

    def resampling(self):

        for i in range(len(self._particles)):
            self._particles[i][0] += np.random.randint(low=-self._resample, high=self._resample+1)
            self._particles[i][1] += np.random.randint(low=-self._resample, high=self._resample+1)


def hsv_color_filter(frame: np.ndarray, upper_range: np.ndarray, lower_range: np.ndarray) -> np.ndarray or None:
    """
    Apply a color filter in the HSV color space.
    :param frame: Frame to apply the filter.
    :param upper_range: a numpy array with the upper limit of the color filter.
    :param lower_range: a numpy array with the lower limit of the color filter.
    :return: filtered frame or None if filter is too little.
    """

    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv_image, lower_range, upper_range)

    mask = cv2.dilate(mask, np.ones((3, 3)), iterations=3)
    mask = cv2.erode(mask, np.ones((3, 3)), iterations=3)

    return (mask / 255) if np.count_nonzero(mask) > 20 else None


def draw_particles(frame: np.ndarray, particles: np.ndarray, particle_size: int, color: (int, int, int), size: int) -> np.ndarray:

    for y, x in particles:
        frame = cv2.rectangle(frame, (x, y), (x + particle_size, y + particle_size), color, size)

    return frame


def main() -> None:
    path = "Secuencia/"

    lower_range, upper_range = np.array([0, 100, 150]), np.array([255, 255, 255])

    particles_size = 35
    num_particles = 40

    filter_particles = ParticlesFilter(num_particles=num_particles, particle_size=particles_size,
                                       image_size=(240, 320), resample=30)

    for num in range(1, 41):

        if num == 1:
            # Initialize the particles
            filter_particles.initialization()

        img = cv2.imread(path + str(num) + ".jpg")
        mask = hsv_color_filter(img, upper_range, lower_range)

        if mask is not None:
            # Evaluation
            filter_particles.evaluation(mask)

            # Estimation
            high_part = filter_particles.estimation()

            # Selection
            filter_particles.selection()

            # Resampling
            filter_particles.resampling()

            draw_particles(img, high_part, particles_size, (0, 0, 255), 3)

            cv2.imshow("Particles Filter", img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
