"""
%---------------------------------------------------------------------------------------------
% For Paper,
% "Surface Similarity Parameter: A new machine learning loss metric for oscillatory spatio-temporal data"
% by Mathies Wedler and Merten Stender and Marco Klein and Svenja Ehlers and Norbert Hoffmann
% Copyright (c) Dynamics Group, Hamburg University of Technology. All rights reserved.
% Licensed under the GPLv3. See LICENSE in the project root for license information.
% Method is based on the work
% "Marc Perlin and Miguel D. Bustamante,
% A robust quantitative comparison criterion of two signals based on the Sobolev norm of their difference,
% Journal of Engineering Mathematics, volume 101,
% DOI 10.1007/s10665-016-9849-7"
%--------------------------------------------------------------------------------------------
"""


from unittest import TestCase
from ssp.ssp_tensorflow import SurfaceSimilarityParameter, SurfaceSimilarityParameterLowPass
import numpy as np
import matplotlib.pyplot as plt


class TestSurfaceSimilarityParameter(TestCase):
    def test_perfect_agreement(self):
        ssp = SurfaceSimilarityParameter()
        x = np.arange(0, 2*np.pi, 0.01)
        y1 = np.sin(x).reshape(1, -1)

        self.assertEqual(ssp(y1, y1).numpy(), 0.0)

    def test_perfect_disagreement(self):
        ssp = SurfaceSimilarityParameter()
        x = np.arange(0, 2*np.pi, 0.01)
        y1 = np.sin(x).reshape(1, -1)
        y2 = np.sin(x + np.pi).reshape(1, -1)

        self.assertEqual(ssp(y1, y2).numpy(), 1.0)

    def test_reduction(self):
        import tensorflow as tf

        x = np.arange(0, 2 * np.pi, 0.01)
        y1 = np.sin(x).reshape(1, -1)

        self.assertGreater(SurfaceSimilarityParameter(reduction=tf.keras.losses.Reduction.NONE)(y1, y1).numpy().ndim, 0)
        self.assertEqual(SurfaceSimilarityParameter()(y1, y1).numpy().ndim, 0)  # default: Reduction.AUTO

    def test_static_filter(self):
        x, y, k = self.get_signal()

        # define ssp with static filter
        ssp = SurfaceSimilarityParameterLowPass(k=k, k_filter=10, lowpass='static')

        # # check if static filter is defined in __init__
        # self.assertIsNotNone(ssp.static_filter)

        # check if ground truth signal is filtered
        self.assertNotEqual(ssp(y, y).numpy(), 0.0)

        # # finally plot filter
        # self.plot_filter(x=x, y=y, ssp=ssp)

    def test_adaptive_filter(self):
        x, y, k = self.get_signal()

        # define ssp with static filter
        ssp = SurfaceSimilarityParameterLowPass(k=k, k_filter=2.5, lowpass='adaptive')

        # check if ground truth signal is filtered
        self.assertNotEqual(ssp(y, y), 0.0)

        # # finally plot filter
        # self.plot_filter(x=x, y=y, ssp=ssp)

    @staticmethod
    def get_signal() -> tuple:
        domain_length = 2 * np.pi
        nx = 1024
        dx = domain_length / nx

        # get space vector x and wave-number vector k
        x = np.arange(0, domain_length, dx)
        k = np.fft.fftfreq(n=len(x), d=dx) * 2 * np.pi

        # define some wave-numbers for test signal
        wave_numbers = np.arange(2, 6, 0.25)

        # generate signal and add noise
        y = np.array(sum([np.sin(k_i * x + phi) for k_i, phi in
                          zip(wave_numbers, 2 * np.pi * np.random.rand(len(wave_numbers)))])).reshape(1, -1)
        y += np.random.rand(len(x))
        return x, y, k

    @staticmethod
    def plot_filter(x, y, ssp):
        y_f = np.fft.fft(y)

        try:
            y_filtered_f = y_f * ssp.static_filter
            adaptive_filter = None
        except TypeError:
            adaptive_filter = ssp.get_adaptive_filter(y_f=y_f)
            y_filtered_f = y_f * adaptive_filter

        fig, ax = plt.subplots(2, 1)
        ax[0].plot(x, y[0], label='ground truth')
        ax[0].plot(x, np.fft.ifft(y_filtered_f)[0].real, label='filtered ground truth')
        ax[0].legend(loc=1)

        ax[1].stem(np.arange(-len(x) // 2, len(x) // 2, 1), np.abs(np.fft.fftshift(y_f[0]) / len(y) * 2), linefmt='tab:blue')
        markerline, _, _, = ax[1].stem(np.arange(-len(x) // 2, len(x) // 2, 1), np.abs(np.fft.fftshift(y_filtered_f[0]) / len(y) * 2), linefmt='tab:orange')

        ax2 = ax[1].twinx()
        ax2.plot(np.fft.fftshift(ssp.k), np.abs(np.fft.fftshift(ssp.static_filter if adaptive_filter is None else adaptive_filter[0])), c='k')

        ax[0].set(xlabel=r'$x$', ylabel=r'$y$')
        ax[1].set(xlabel=r'$k$', ylabel=r'$S(k)$', xlim=(0, 15))
        plt.setp(markerline, markersize=4)
        plt.suptitle(f'SSP with {"adaptive" if ssp.static_filter is None else "static"} low pass filter\n'
                     f'ML model fits filtered data when employed as loss function')
        plt.show()
