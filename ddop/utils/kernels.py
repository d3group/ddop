import math


class Kernel:
    def __init__(self, kernel_type, kernel_weight):
        self.kernel_type = kernel_type
        self.kernel_weight = kernel_weight
        self.kernel_type = "gaussian"

    def __uniform_kernel(self, u):
        if (u / self.kernel_weight) <= 1:
            k_w = 0.5 / self.kernel_weight
        else:
            k_w = 0
        return k_w

    def __gaussian_kernel(self, u):
        k_w = 1 / math.sqrt(2 * math.pi) * math.pow(math.e, -0.5 * math.pow(u / self.kernel_weight, 2))
        return k_w

    def get_kernel_output(self, u):
        if self.kernel_type == "uniform":
            return self.__uniform_kernel(u)

        elif self.kernel_type == "gaussian":
            return self.__gaussian_kernel(u)

        else:
            raise ValueError("Kernel type must be one of ‘uniform’, ‘gaussian’")
