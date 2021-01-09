import math


class Kernel:
    def __init__(self, kernel_type, kernel_bandwidth):
        self.kernel_type = kernel_type
        self.kernel_bandwidth = kernel_bandwidth

    def _uniform_kernel(self, u):
        if (u / self.kernel_bandwidth) <= 1:
            k_w = 0.5 / self.kernel_bandwidth
        else:
            k_w = 0
        return k_w

    def _gaussian_kernel(self, u):
        k_w = 1  * math.pow(math.e, -0.5 * math.pow(u / self.kernel_bandwidth, 2))
        return k_w/self.kernel_bandwidth

    def get_kernel_output(self, u):
        if self.kernel_type == "uniform":
            return self._uniform_kernel(u)

        elif self.kernel_type == "gaussian":
            return self._gaussian_kernel(u)

        else:
            raise ValueError("Kernel type must be one of ‘uniform’, ‘gaussian’")
