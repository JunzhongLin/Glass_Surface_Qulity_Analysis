import numpy as np
import pandas as pd
import glob
import pickle
from scipy.signal import find_peaks, savgol_filter
import matplotlib.pyplot as plt
import pickle
from afm_func_1d import *   # this is a bad practice
import afm_func_1d
from collections import namedtuple
from functools import lru_cache

Sample = namedtuple('Sample', 'sample_name roughness_param target_prop')
# here the target_prop means the target property we are focusing on, in my case, it is the ball on ring strength


class RoughnessParam:

    # Generate the data pool for the predictors (e.g. roughness parameters) from AFM measurements, and
    # the data pool for the target property from the input directly. the roughness parameters were extracted
    # from the line profiles, which are obtained from the AFM measurement

    # the line profile were selected by a scan window with stride through the AFM measurements

    def __init__(self, path, hyper_param, name_list):
        '''
        :param path: define the path for the afm data and target property (excel file).
        the firt three columns of the excel file should contain x, y, z value from a afm measurement.
        such data can be obtained by processing the raw afm data using an open source software called
        'Gwyddion'. Please refer to 'http://gwyddion.net'

        :param hyper_param: A dictionary defining the parameter for the generation process of line
        profiles from AFM data.
            size : the number of pixels along x axis and y axis from the AFM measuremnt, (e.g. 256)
            window: the number rows selected to calculate the mean value as the line profile
            stride: number of rows between individual window to be discarded in order to ensure the
            independency
            use_smooth: (boolean value). whether to do the smoothing of line profile or not

        :param name_list: name list of roughness parameters (e.g. ['Rt', 'Rp', 'Rv', 'Rzjis'])
        please NOTE that those names should match up the functions defined in the afm_func_1d.py file
        for the calculation of roughness parameters
        '''
        data = pd.read_excel(path)
        self.data = data.z.values.reshape(hyper_param['size'], hyper_param['size'])
        self.hyper_param = hyper_param
        self.target_property_data = data.iloc[:, 3:]
        self.sample_name = path.split('/')[-1]
        self.size_x = hyper_param['size']   # pixels along x axis for the AFM measurement
        self.size_y = hyper_param['size']   # pixels along y axis for the AFM measurement
        self.stride = hyper_param['stride']
        self.window = hyper_param['window']
        self.use_smooth = hyper_param['use_smooth']
        self.roughness_param_dict = {}
        self.target_property_dict = {}
        self.param_name_list = name_list
        self.profile_pool = self._gen_profiles()
        self.main()

    # The AFM measurement will provide a 3 dimentional result, the x and y axes define the position and the z
    # axis define the height. Here I want to use the AFM result to generate a profile pool, which contains only
    # line profiles inside.
    def _gen_profiles(self,):

        profile_pool = np.zeros([2*self.size_x//self.stride, self.size_y])
        t_data = self.data.transpose()
        for i in range(self.size_x//self.stride):
            # stride is used here to discard some data in order to ensure the independancy
            profile_pool[i, :] = self.data[i*self.stride:self.window+i*self.stride, :].mean(axis=0)
            profile_pool[i+self.size_x//self.stride, :] = t_data[i*self.stride:self.window+i*self.stride, :].mean(axis=0)

        # line profile smoothing
        smooth_profile_pool = savgol_filter(profile_pool, window_length=self.hyper_param['window_length'],
                                            polyorder=self.hyper_param['polyorder'])
        return profile_pool if not self.use_smooth else smooth_profile_pool

    def main(self):
        for roughness_param in self.param_name_list:
            cal_method = getattr(afm_func_1d, roughness_param)
            self.roughness_param_dict.setdefault(roughness_param, cal_method(self.profile_pool, **self.hyper_param))
        target_prop_names = self.target_property_data.columns
        for name in target_prop_names:
            self.target_property_dict.setdefault(name,
                                                 self.target_property_data.loc[:, name].dropna().values)

        return self.roughness_param_dict, self.target_property_dict


class DataSet:

    # Create a dataset using all the excel files within the same folder

    def __init__(self, hyper_param, name_list):

        '''
        :param hyper_param: same definition from class RoughnessParam
        :param name_list: same definition from class RoughnessParam
        '''

        self.dataset = []
        self.hyperparam_for_roughness = hyper_param
        self.param_name_list = name_list

    def main(self, afm_file_path):

        '''
        :param afm_file_path: define the folder contains all the excel files for each sample
        :return: a list of turples, each turple is defined for each sample, which contains the sample
        name, roughness parameters and target properties
        '''

        afm_file_list = glob.glob(afm_file_path) # obtain the list of excel files in the folder
        for file in afm_file_list:
            temp = RoughnessParam(file, self.hyperparam_for_roughness, self.param_name_list)
            sample = Sample(temp.sample_name, temp.roughness_param_dict, temp.target_property_dict)
            self.dataset.append(sample)
        return self.dataset


# if __name__ == '__main__':
#
#     afm_folder = r'./data/test/*'
#     param_name_list = [
#         'Rt', 'Rp', 'Rv', 'Rzjis', 'Ra', 'Rq', 'Rsk', 'Rku', 'delta_ANAM'
#     ]
#     hyperparam_for_roughness = {
#         'tau_list': np.arange(1, 10),
#         'alpha': 2,
#         'window_length': 31,
#         'polyorder': 5,
#         'size': 256,
#         'stride': 32,
#         'window': 4,
#         'use_smooth': True
#     }
#
#     dset_1 = DataSet(hyperparam_for_roughness, param_name_list)
#     data = dset_1.main(afm_folder)