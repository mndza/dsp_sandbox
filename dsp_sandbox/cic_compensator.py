import numpy as np
from scipy.special import zeta

from dsp_sandbox.fir import FIRFilter
from dsp_sandbox.types.fixed_point import Q


class MaximallyFlatCICCompensator(FIRFilter):
    def __init__(self, stages, rate, ntaps, width_in, width_out):
        taps = self.get_taps(stages, rate, ntaps)
        taps /= np.max(taps)
        super().__init__(taps, Q(width_in, 0), Q(width_out, 0), shape_taps=Q(2, 10))

    @staticmethod
    def get_taps(N, R, ntaps):
        """
        Reference:
        "Closed-Form Design of CIC Compensators Based on Maximally Flat Error Criterion"

        Based on ciccompmf (MATLAB script)
        """
        def berno(k):
            B = [1/6, -1/30, 1/42, -1/30, 5/66, -691/2730, 7/6, -3617/510, 43867/798]
            return B[k-1]

        def berno(k):
            B2k = 2 * (-1)**(k-1) * np.prod(np.arange(1, 2*k+1)) / (2*np.pi)**(2*k) * zeta(2*k)
            return B2k
        
        ntaps_h = ntaps // 2

        if ntaps % 2 == 1:  # odd ntaps
            u = np.arange(1, ntaps_h + 1).reshape(-1, 1) * np.ones((1, ntaps_h))
            A = 2 * (-1)**u * u.transpose()**(2*u)
            b = np.zeros((ntaps_h, 1))
            for u in range(1, ntaps_h + 1):
                for q in range(1, u+1):
                    b[u-1] += (2*q - 1) * (N * abs(berno(u-q+1)) / 2 / (u-q+1) * (1 - 1 / R**(2*(u-q+1)) ) )**q
            a = np.linalg.solve(A, b).flatten()
            a0 = 1 - 2*np.sum(a)
            h = np.concatenate((a[::-1], [a0], a))
        else:  # even ntaps
            u = np.arange(1, ntaps_h + 1).reshape(-1, 1) * np.ones((1, ntaps_h))
            A = 2 * (-1)**(u-1) * (u.transpose()-1/2)**(2*u-2)
            b = np.concatenate((np.ones((1, 1)), np.zeros((ntaps_h-1, 1))))
            for u in range(2, ntaps_h + 1):
                for q in range(1, u):
                    b[u-1] += (2*q - 1) * (N * abs(berno(u-q)) / 2 / (u-q) * (1 - 1 / R**(2*(u-q)) ) )**q
            a = np.linalg.solve(A, b).flatten()
            h = np.concatenate((a[::-1], a))
            
        return h
