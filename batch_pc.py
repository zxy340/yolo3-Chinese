from tkinter import CENTER
from turtle import distance
import numpy as np
import matplotlib.pyplot as plt
import os
import doatools.doatools.model as model
import doatools.doatools.plotting as doaplt
from doatools.doatools.plotting import plot_array, plot_spectrum
import doatools.doatools.estimation as estimation
from doatools.doatools.estimation.core import SpectrumBasedEstimatorBase, ensure_covariance_size
from doatools.doatools.model.sources import FarField2DSourcePlacement
# import plotly.graph_objects as go
from scipy.io import savemat
# import cupy as cp
import multiprocessing
import cv2

# from raw_data_extract import *

def IWR6843TDM_lazy(dfile, num_frame, num_chirp, num_Tx, num_channel = 4, num_sample=256):
    """ Lazy load adc data

    Parameters
    ----------
    dfile : str
        an reorganized adc_data file
    num_frame : int
    num_chirp : int
    num_Tx: int
    num_channel: int
    num_sample : int

    Returns
    -------
    np.memmap
        the I, Q data of each sample are not reordered
    """
    if num_frame == -1:
        num_frame = os.path.getsize(dfile) // num_chirp // num_Tx // num_channel // num_sample // 2 // 2
    return np.memmap(dfile, mode='r', dtype=np.int16, shape=(num_frame, num_chirp, num_Tx, num_channel, num_sample * 2))


def reorder_IQ(d):
    """ reorder the IIQQ format of data to I+1j*Q, the returned data is in memory

    Parameters
    ----------
    d : np.memmap
        the data loaded by awr1642_lazy or IWR6843TDM_lazy

    Returns
    -------
    ndarray
    """
    shape = list(d.shape)
    shape[-1] = -1
    shape.append(4)
    d = d.reshape(shape)
    d = d[..., :2] + 1j * d[..., 2:]
    d = d.reshape(shape[:-1])
    return d

def phase_flip(d):
    """
    add -180 degree phase inversion to the upside-down Rx on IWR6843ISK-ODS

    Parameters
    ----------
    d : np.ndarray
        the reordered IQ data

    Returns
    -------
    ndarray

    """
    rotation = np.array([0, -np.pi, -np.pi, 0, 0, -np.pi, -np.pi, 0, 0, -np.pi, -np.pi, 0]).reshape(3, 4)
    # rotation = -rotation
    # rotation = rotation + np.pi
    # rotation = -(rotation + np.pi)
    return np.fft.ifft(np.fft.fft(d, axis=-1) * np.exp(1j*rotation)[...,None])  # broardcase: (..., Tx, Channel, Sample) * (3, 4, None)


wavelength = 3e8 / 60e9
spacing = wavelength / 2.0
vir_ant = np.array([
    [0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [3, 2, 2, 3, 3, 2, 2, 3, 1, 0, 0, 1]
]).T * spacing
vir_ant[:, 2] = -vir_ant[:, 2]  # lefthand to righthand coor
AntennaArr = model.ArrayDesign(vir_ant, name='IWR6843ISK-ODS')


class MVDR(SpectrumBasedEstimatorBase):
    def __init__(self, array, wavelength, search_grid, **kwargs):
        super().__init__(array, wavelength, search_grid, **kwargs)

    @staticmethod
    def _cov(signal):
        """
        get signal's covariance matrix

        Parameters
        ----------
        signal: np.ndarray
            c by t matrix where c is the channel and t is the (slow) time

        Returns
        -------
        np.ndarray
            covariance matrix
        """
        return signal @ (signal.conj().T) / signal.shape[-1]

    def estimate(self, signal):
        """
        estimate the MVDR spectrum

        Parameters
        ----------
        signal: np.ndarray
            c by t matrix where c is the channel and t is the (slow) time

        Returns
        -------
        sp: np.ndarray
            MDDR spectrum. The size and resolution is determined by search_grid
        """
        R = MVDR._cov(signal)
        ensure_covariance_size(R, self._array)

        sp = estimation.beamforming.f_mvdr(self._get_atom_matrix(), R)
        sp = sp.reshape(self._search_grid.shape)
        return sp

    def dbf(self, signal, source):
        """
        digital beamforming using MVDR

        Parameters
        ----------
        signal: np.ndarray
            c by t matrix where c is the channel and t is the (slow) time
        source: doatools.model.sources.SourcePlacement
            sources where MVDR perform dbf

        Returns
        -------
        np.ndarray
            beamformed signal, shape=(*grid.shape, t)
        """
        R = MVDR._cov(signal)
        ensure_covariance_size(R, self._array)

        sv = self._array.steering_matrix(
            source, self._wavelength,
            perturbations='known'
        )
        R_invSV = np.linalg.lstsq(R, sv, None)[0]
        spatial_filter = R_invSV / (sv.conj() * R_invSV)
        spatial_filter = spatial_filter.conj()  # channel, source
        ret = np.einsum("ct, cs->st", signal, spatial_filter)  # mult each channel with the weight and sum up, here t is slow time
        return ret

    def dbf_grid(self, signal, grid=None):
        """
        grid digital beamforming using MVDR

        Parameters
        ----------
        signal: np.ndarray
            c by t matrix where c is the channel and t is the (slow) time
        grid: doatools.estimation.grid.SearchGrid
            a search grid where MVDR perform dbf, the default DoA grid is used if grid is None

        Returns
        -------
        np.ndarray
            beamformed signal, shape=(*grid.shape, t)
        """

        if grid == None:
            grid = self._search_grid
        ret = self.dbf(signal, grid.source_placement)
        ret = ret.reshape(*grid.shape, ret.shape[-1])  # source placement is flattened, so reshape is required
        return ret

def cloudpoint(data, num_chirp, num_Tx, num_channel, num_sample, az_start, az_end, az_step, el_start, el_end, el_step):
    data = reorder_IQ(data)
    data = phase_flip(data)

    # range fft
    data = np.fft.fft(data, axis=-1)
    data = data.reshape(num_chirp, num_Tx*num_channel, num_sample)

    az_size = int((az_end-az_start)/az_step)
    el_size = int((el_end-el_start)/el_step)
    grid = estimation.FarField2DSearchGrid(start=(az_start,el_start),stop=(az_end,el_end), unit='deg', size=(az_size,el_size))
    estimator = MVDR(AntennaArr, wavelength, grid)


    point_cloud = np.zeros((az_size, el_size, num_sample))
    for s in range(num_sample):
        point_cloud[..., s] = estimator.estimate(data[..., s].T)
    return estimator, point_cloud

def data_process(data):
    az_start, az_end =  20, 141 # end not included
    az_step = 1
    el_start, el_end = -80, 71
    el_step = 1
    num_chirp=128
    num_Tx=3
    num_channel=4
    num_sample=256

    #heatmap
    center_count = 20
    surround_center_count = 1000

    est, p = cloudpoint(data, num_chirp=num_chirp, num_Tx=num_Tx, num_channel=num_channel, num_sample=num_sample, az_start=az_start, az_end=az_end, az_step=az_step, el_start=el_start, el_end=el_end, el_step=el_step)
    idx = np.argsort(p, axis=None)[::-1]
    az, el, d = np.unravel_index(idx, p.shape)
    az = az*az_step + az_start
    el = el*el_step + el_start
    z = d * np.sin(el*np.pi/180.0)
    x = d * np.cos(el*np.pi/180.0) * np.cos(az*np.pi/180.0)
    y = d * np.cos(el*np.pi/180.0) * np.sin(az*np.pi/180.0)
    power = np.log(p[((az-az_start)/az_step).astype(int), ((el-el_start)/el_step).astype(int), d])
    power_idx = np.argsort(power)[::-1]
    x = x.reshape((-1,1))
    y = y.reshape((-1,1))
    z = z.reshape((-1,1))
    points = np.concatenate((x,y,z),axis=1)

    power = power[power_idx]
    points = points[power_idx]
    y = points[:,1]
    z = points[:,2]
    x = points[:,0]
    matching_idx = np.where((np.abs(x)<25)&(y>20) & (y<30) & (np.abs(z)<25))
    points = points[matching_idx]
    cluster_count = 0
    center = points[0]
    cluster = []

    while cluster_count != center_count:
        distances = np.sqrt(np.sum(np.power((points - center),2)*np.array([1,4,1]),axis=1))
        distances_idx = np.argsort(distances,axis=None)[:surround_center_count]

        temp_cluster = points[distances_idx]
        temp_cluster_power = power[distances_idx].reshape((-1,1))
        temp_cluster = np.concatenate((temp_cluster,temp_cluster_power),axis=1)

        indices = np.setxor1d(np.arange(points.shape[0]),distances_idx)
        points = points[indices]
        power = power[indices]

        power_idx = np.argsort(power)[::-1]
        points = points[power_idx]
        power = power[power_idx]
        center = points[0]

        cluster.append(temp_cluster)
        cluster_count +=1
    cluster = np.concatenate(cluster,axis=0)

    ax = plt.subplot(1, 1, 1)
    plt.xlim((-25, 25))
    plt.ylim((-25, 25))
    colors = cluster[:, 3]
    ax.scatter(-cluster[:, 2], -cluster[:, 0],
               cmap='jet',
               c=colors,
               s=12,
               linewidth=0,
               alpha=1,
               marker=".")
    plt.savefig('./temp.jpg')
    plt.close()
    image = cv2.imread('./temp.jpg')
    image = cv2.resize(image, (416, 416))
    cv2.imwrite('./realtime/mm_image.jpg', image)

    return './realtime/mm_image.jpg'
