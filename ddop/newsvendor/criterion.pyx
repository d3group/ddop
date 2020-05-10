from sklearn.tree._criterion cimport Criterion
from sklearn.tree._criterion cimport SIZE_t, DOUBLE_t
from sklearn.tree._utils cimport sizet_ptr_to_ndarray
from sklearn.tree._utils cimport safe_realloc
from libc.stdlib cimport calloc
from libc.string cimport memcpy
from libc.string cimport memset
from libc.math cimport fabs
import numpy as np
cimport numpy as np
from libc.stdio cimport printf

np.import_array()


cdef class NewsvendorCriterion(Criterion):
    """Newsvendor impurity criterion, which minimizes the following loss function:
        Loss(q(x),D) = sum_{i=1}^{N} co(q(x)-d)^+ + cu(d-q(x))^+
        Here, excess quantity (i.e., if (q(x)-d) > 0) is considered with co in the
        loss function, whereas missing quantities ((q(x)-d) < 0) are weighted with co.

        The code was inspired by:
        <https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/tree/_criterion.pyx>`_.
    """
    #cdef double cu
    cdef  SIZE_t* cu
    cdef  SIZE_t* co

    def __cinit__(self, SIZE_t n_outputs, SIZE_t n_samples, np.ndarray[SIZE_t, ndim=1] cu,
                  np.ndarray[SIZE_t, ndim=1] co):
        """Initialize parameters for this criterion.
        Parameters
        ----------
        n_outputs : SIZE_t
            The number of targets to be predicted
        n_samples : SIZE_t
            The total number of samples to fit on
        cu : SIZE_t
            The underage costs per unit.
        co : SIZE_t
            The overage costs per unit:
        """

        self.cu = NULL
        self.co = NULL

        # Default values
        self.sample_weight = NULL

        self.samples = NULL
        self.start = 0
        self.pos = 0
        self.end = 0

        self.n_outputs = n_outputs
        self.n_samples = n_samples
        self.n_node_samples = 0
        self.weighted_n_node_samples = 0.0
        self.weighted_n_left = 0.0
        self.weighted_n_right = 0.0

        # Allocate accumulators. Make sure they are NULL, not uninitialized,
        # before an exception can be raised (which triggers __dealloc__).
        self.sum_total = NULL
        self.sum_left = NULL
        self.sum_right = NULL

        safe_realloc(&self.cu, n_outputs)
        safe_realloc(&self.co, n_outputs)

        # Allocate memory for the accumulators
        self.sum_total = <double*> calloc(n_outputs, sizeof(double))
        self.sum_left = <double*> calloc(n_outputs, sizeof(double))
        self.sum_right = <double*> calloc(n_outputs, sizeof(double))

        if (self.sum_total == NULL or
                self.sum_left == NULL or
                self.sum_right == NULL):
            raise MemoryError()

        cdef SIZE_t k = 0
        for k in range(n_outputs):
            self.cu[k] = cu[k]
            self.co[k] = co[k]


    def __reduce__(self):
        return type(self), (self.n_outputs, self.n_samples, sizet_ptr_to_ndarray(self.cu, self.n_outputs),
                            sizet_ptr_to_ndarray(self.co, self.n_outputs)), self.__getstate__()

    cdef int init(self, const DOUBLE_t[:, ::1] y, DOUBLE_t* sample_weight,
                  double weighted_n_samples, SIZE_t* samples, SIZE_t start,
                  SIZE_t end) nogil except -1:
        """Initialize the criterion at node samples[start:end] and
           children samples[start:start] and samples[start:end]."""
        # Initialize fields
        self.y = y
        self.sample_weight = sample_weight
        self.samples = samples
        self.start = start
        self.end = end
        self.n_node_samples = end - start
        self.weighted_n_samples = weighted_n_samples
        self.weighted_n_node_samples = 0.

        cdef SIZE_t i
        cdef SIZE_t p
        cdef SIZE_t k
        cdef DOUBLE_t y_ik
        cdef DOUBLE_t w_y_ik
        cdef DOUBLE_t w = 1.0

        memset(self.sum_total, 0, self.n_outputs * sizeof(double))

        for p in range(start, end):
            i = samples[p]

            if sample_weight != NULL:
                w = sample_weight[i]

            for k in range(self.n_outputs):
                y_ik = self.y[i, k]
                w_y_ik = w * y_ik
                self.sum_total[k] += w_y_ik

            self.weighted_n_node_samples += w

        # Reset to pos=start
        self.reset()
        return 0

    cdef int reset(self) nogil except -1:
        """Reset the criterion at pos=start."""
        cdef SIZE_t n_bytes = self.n_outputs * sizeof(double)
        memset(self.sum_left, 0, n_bytes)
        memcpy(self.sum_right, self.sum_total, n_bytes)

        self.weighted_n_left = 0.0
        self.weighted_n_right = self.weighted_n_node_samples
        self.pos = self.start
        return 0

    cdef int reverse_reset(self) nogil except -1:
        """Reset the criterion at pos=end."""
        cdef SIZE_t n_bytes = self.n_outputs * sizeof(double)
        memset(self.sum_right, 0, n_bytes)
        memcpy(self.sum_left, self.sum_total, n_bytes)

        self.weighted_n_right = 0.0
        self.weighted_n_left = self.weighted_n_node_samples
        self.pos = self.end
        return 0

    cdef int update(self, SIZE_t new_pos) nogil except -1:
        """Updated statistics by moving samples[pos:new_pos] to the left."""

        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right
        cdef double* sum_total = self.sum_total

        cdef double* sample_weight = self.sample_weight
        cdef SIZE_t* samples = self.samples

        cdef SIZE_t pos = self.pos
        cdef SIZE_t end = self.end
        cdef SIZE_t i
        cdef SIZE_t p
        cdef SIZE_t k
        cdef DOUBLE_t w = 1.0

        # Update statistics up to new_pos
        #
        # Given that
        #           sum_left[x] +  sum_right[x] = sum_total[x]
        # and that sum_total is known, we are going to update
        # sum_left from the direction that require the least amount
        # of computations, i.e. from pos to new_pos or from end to new_pos.

        if (new_pos - pos) <= (end - new_pos):
            for p in range(pos, new_pos):
                i = samples[p]

                if sample_weight != NULL:
                    w = sample_weight[i]

                for k in range(self.n_outputs):
                    sum_left[k] += w * self.y[i, k]

                self.weighted_n_left += w
        else:
            self.reverse_reset()

            for p in range(end - 1, new_pos - 1, -1):
                i = samples[p]

                if sample_weight != NULL:
                    w = sample_weight[i]

                for k in range(self.n_outputs):
                    sum_left[k] -= w * self.y[i, k]

                self.weighted_n_left -= w

        self.weighted_n_right = (self.weighted_n_node_samples -
                                 self.weighted_n_left)
        for k in range(self.n_outputs):
            sum_right[k] = sum_total[k] - sum_left[k]

        self.pos = new_pos
        return 0


    cdef double node_impurity(self) nogil:
        """Evaluate the impurity of the current node, i.e. the impurity of
           samples[start:end]"""

        cdef DOUBLE_t* sample_weight = self.sample_weight
        cdef SIZE_t* samples = self.samples
        cdef SIZE_t i, p, k
        cdef DOUBLE_t w = 1.0
        cdef DOUBLE_t impurity = 0.0

        cdef double* sum_total = self.sum_total

        cdef SIZE_t* cu = self.cu
        cdef SIZE_t* co = self.co

        for k in range(self.n_outputs):

            for p in range(self.start, self.end):
                i = samples[p]

                if sample_weight != NULL:
                    w = sample_weight[i]

                if self.y[i, k] - sum_total[k] / self.weighted_n_node_samples > 0.0:
                    impurity += fabs(self.y[i, k] - sum_total[k] / self.weighted_n_node_samples) * cu[k]
                else:
                    impurity += fabs(self.y[i, k] - sum_total[k] / self.weighted_n_node_samples) * co[k]

        return impurity

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        """Evaluate the impurity in children nodes, i.e. the impurity of the
           left child (samples[start:pos]) and the impurity the right child
           (samples[pos:end])."""

        cdef DOUBLE_t* sample_weight = self.sample_weight
        cdef SIZE_t* samples = self.samples
        cdef SIZE_t pos = self.pos
        cdef SIZE_t start = self.start
        cdef SIZE_t end = self.end

        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right
        cdef DOUBLE_t y_ik

        cdef double impurity_left_temp = 0.0
        cdef double impurity_right_temp = 0.0

        cdef SIZE_t i
        cdef SIZE_t p
        cdef SIZE_t k
        cdef DOUBLE_t w = 1.0

        cdef SIZE_t* cu = self.cu
        cdef SIZE_t* co = self.co

        # left child
        for k in range(self.n_outputs):
            for p in range(start, pos):
                i = samples[p]

                if sample_weight != NULL:
                    w = sample_weight[i]

                if self.y[i, k] - sum_left[k] / self.weighted_n_left > 0.0:
                    impurity_left_temp += fabs(self.y[i, k] - sum_left[k] / self.weighted_n_left) * cu[k]
                else:
                    impurity_left_temp += fabs(self.y[i, k] - sum_left[k] / self.weighted_n_left) * co[k]

        # right child
        for k in range(self.n_outputs):
            for p in range(pos, end):
                i = samples[p]

                if sample_weight != NULL:
                    w = sample_weight[i]

                if self.y[i, k] - sum_right[k] / self.weighted_n_right > 0.0:
                    impurity_right_temp += fabs(self.y[i, k] - sum_right[k] / self.weighted_n_right) * cu[k]
                else:
                    impurity_right_temp += fabs(self.y[i, k] - sum_right[k] / self.weighted_n_right) * co[k]

        impurity_left[0] = impurity_left_temp
        impurity_right[0] = impurity_right_temp

    cdef void node_value(self, double* dest) nogil:
        """Compute the node value of samples[start:end] into dest."""

        cdef SIZE_t k

        for k in range(self.n_outputs):
            dest[k] = self.sum_total[k] / self.weighted_n_node_samples