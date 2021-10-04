import numpy as np
import pandas as pd
import statsmodels as sm
import distutils
from dset import DataSet
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from collections import namedtuple
from scipy.stats import pearsonr

res_tuple = namedtuple('result_miu_p', 'name miu ci_miu_l ci_miu_r p_miu ci_p_l ci_p_r')


def linear_func(x, a, b):
    return a*x+b


class Regression:

    # To perform bootstrapped linear regression

    def __init__(self, folder, hyper_param, name_list,):
        '''

        :param folder: the folder contains all the excel files for the afm xyz data and target property
        :param hyper_param: parameter used to obtain the line profile from AFM measurments, calculating
        the specific roughness parameter, like delta_ANAM (fractal dimension)
        :param name_list: the list of names for the roughness parameters which we want to involve into
        the analysis. It is same with the one in the class of RoughnessParam

        '''
        self.folder = folder
        self.hyper_param = hyper_param
        self.name_list = name_list
        self.data_pool = None
        self.get_full_data()

    def get_full_data(self):

        # To obtain the full dataset contains the pool for roughness parameter and target property
        # for each sample
        dataset = DataSet(self.hyper_param, self.name_list)
        self.data_pool = dataset.main(self.folder)
        return self.data_pool

    def bootstrap_sampling(self, param_name, prop_name, bp_size=40000,):
        '''
        To perform the bootstrap sampling with replacement on the dataset generated previously

        :param param_name: name of individual roughness parameter presented in the name_list
        :param prop_name: name of target property (e.g. BOR,), which should match up with the name
        in the excel file
        :param bp_size: times of boostrap sampling

        :return:
            x: the roughness parameter
            y: target property
        # each row represents the individual bootstrap sampling, each column represents
        # the mean value of roughness parameter for a specific sample obtained by
        # bootstrapping.
        '''
        sample_list = []
        x = np.zeros([bp_size, len(self.data_pool)])
        y = np.zeros([bp_size, len(self.data_pool)])
        # each row represents the individual bootstrap sampling, each column represents
        # the mean value of roughness parameter for a specific sample obtained by
        # bootstrapping.

        for ndx, sample in enumerate(self.data_pool):
            sample_list.append(sample.sample_name)
            param_pool = np.array([self.data_pool[ndx].roughness_param[param_name],]*bp_size)
            target_pool = np.array([self.data_pool[ndx].target_prop[prop_name],]*bp_size)

            # bootstrap sampling
            param_bp_rand_pool = np.random.randint(param_pool.shape[1], size=param_pool.shape)
            target_bp_rand_pool = np.random.randint(target_pool.shape[1], size=target_pool.shape)
            x[:, ndx] = np.array([param_pool[i][param_bp_rand_pool[i]].mean() for i in range(bp_size)])
            y[:, ndx] = np.array([target_pool[i][target_bp_rand_pool[i]].mean() for i in range(bp_size)])
        return x, y

    def uni_var_reg(self, X, Y):

        '''

        :param X: the output x from bootstrap_sampling function
        :param Y: the output y from bootstrap_sampling function

        :return: a list for slope obtained by regression, and,
                 a list for pearson r value obtained by regression
                 the length of each list is equal to the size of boostrap sampling
        '''

        slope_list = []
        r_value_list = []
        for x, y in zip(X, Y):
            popt, pcov = curve_fit(linear_func, xdata=x, ydata=y)
            slope_list.append(popt)
            r_value = pearsonr(x, y)
            r_value_list.append(r_value)
        return slope_list, r_value_list

    def pi_ci(self, coeff):

        '''

        :param coeff: the coefficient obtained by linear regression, such as the slope, pearson r value

        :return: a list contains named tuple, each tuple represents the regression analysis results
        for each roughness parameter
        '''

        res_list = []
        for name, s in zip(self.name_list, coeff):
            miu = s[:, 0].mean() # mean value of the coefficient
            s_sorted = np.sort(s[:, 0])
            ci_miu_l = s_sorted[int(s_sorted.size*0.05)] # 90% confident interval
            ci_miu_r = s_sorted[int(s_sorted.size*0.95)] # 90% confident interval

            p_sorted = np.abs(s_sorted/(ci_miu_l-ci_miu_r)) # normalized coefficient
            p_miu = p_sorted.mean()
            ci_p_l = p_sorted[int(p_sorted.size*0.05)]
            ci_p_r = p_sorted[int(p_sorted.size*0.95)]
            res_list.append(res_tuple(name, miu,  ci_miu_l, ci_miu_r, p_miu, ci_p_l, ci_p_r))

        return res_list

    def res_visualization(self):
        '''
        visualize the results from regression analysis
        :return:
        '''

        fig_1, ax_1 = plt.subplots(1, 1)

        for slope, name in zip(self.s_values, self.name_list):
            ax_1.hist(slope[:, 0], bins=500, alpha=0.8, label=name, histtype='step', )
            ax_1.legend()
            ax_1.set_xlabel('slopes determined from linear regression', fontsize=14)
            ax_1.set_ylabel('Number of observed Boostrap simulation', fontsize=14)

        fig_1.savefig('./figures/res/slope.png')

        fig_2, ax_2 = plt.subplots(1, 1)

        for r, name in zip(self.r_values, self.name_list):
            ax_2.hist(r[:, 0], bins=500, alpha=0.8, label=name, histtype='step', )
            ax_2.legend()
            ax_2.set_xlabel('pearson r value determined from linear regression', fontsize=14)
            ax_2.set_ylabel('Number of observed Boostrap simulation', fontsize=14)

        fig_2.savefig('./figures/res/r.png')



    def data_visualization(self):

        '''
        visualize the roughness parameter for each sample
        '''

        for param_name in self.name_list:
            data_temp = np.zeros([16, len(self.data_pool)])
            x = [i + 1 for i in range(data_temp.shape[1])]
            id_list = []
            for num, sample in enumerate(self.data_pool):
                data_temp[:, num] = sample.roughness_param[param_name]
                id_list.append(sample.sample_name[:2])
            fig, ax = plt.subplots(1, 1)
            ax.boxplot(data_temp[:, :],
                       vert=True,  # vertical box alignment
                       patch_artist=False,  # fill with color
                       # notch=True,       # notch types
                       labels=id_list,
                       positions=x
                       )
            ax.set_ylabel(param_name, fontsize=13)
            ax.set_xlabel('Sample ID', fontsize=13)
            fig.savefig('./figures/data/{}.png'.format(param_name))


    def main(self):
        '''

        :return:
            slope values, r_values: the result for the regression analysis for each data obtained from
            bootstrapping
            X_s, Y_s : standard scaled for the regression analysis
            slope_ci_pi, r_ci_pi: output from pi_ci function
        '''
        slope_values = []
        r_values = []
        X_s = []
        Y_s = []

        prop_name = 'BOR'

        for param_name in self.name_list:

            X, Y = self.bootstrap_sampling(param_name, prop_name)
            X_hat = (X-X.mean())/X.std() # standard scaling
            Y_hat = (Y-Y.mean())/Y.std() # standard scaling
            slope_temp = np.array(self.uni_var_reg(X_hat, Y_hat)[0])
            r_temp = np.array(self.uni_var_reg(X_hat, Y_hat)[1])
            slope_values.append(slope_temp)
            r_values.append(r_temp)
            X_s.append(X_hat)
            Y_s.append(Y_hat)

        slope_ci_pi = self.pi_ci(slope_values)
        r_ci_pi = self.pi_ci(r_values)

        self.s_values = slope_values
        self.r_values = r_values
        self.slope_ci_pi = slope_ci_pi
        self.r_ci_pi =r_ci_pi

        return slope_values, r_values, X_s, Y_s, slope_ci_pi, r_ci_pi


if __name__ == '__main__':

    afm_folder = r'./data/AS87/*'

    param_name_list = [
        'Rt', 'Rp', 'Rv', 'Rzjis', 'Ra', 'Rq', 'Rsk', 'Rku', 'delta_ANAM'
    ]
    hyperparam_for_roughness = {
        'tau_list': np.arange(1, 10),
        'alpha': 2,
        'window_length': 31,
        'polyorder': 5,
        'size': 256,
        'stride': 32,
        'window': 4,
        'use_smooth': False
    }

    test = Regression(afm_folder, hyperparam_for_roughness, param_name_list)
    s_values, r_values, X, Y, s_ci_pi, r_ci_pi = test.main()
    test.data_visualization()
    test.res_visualization()


'''
    fig_1, ax_1 = plt.subplots(1, 1)
    for slope, name in zip(res, param_name_list):
        ax_1.hist(slope[:, 0], bins=500, alpha=0.8, label=name, histtype='step',)
        ax_1.legend()
        ax_1.set_xlabel('slopes determined from linear regression', fontsize=14)
        ax_1.set_ylabel('Number of observed Boostrap simulation', fontsize=14)


    fig_4, ax_4 = plt.subplots(1, 1)

    for r, name in zip(r_values, param_name_list):
        ax_4.hist(r[:, 0], bins=500, alpha=0.8, label=name, histtype='step',)
        ax_4.legend()
        ax_4.set_xlabel('pearson r value determined from linear regression', fontsize=14)
        ax_4.set_ylabel('Number of observed Boostrap simulation', fontsize=14)


    # fig_2, ax_2 = plt.subplots(1, 1)
    # x = param_name_list
    # y = np.array([res.p_miu for res in res_ci_pi])
    # e = np.array([np.abs(res.ci_p_l-res.ci_p_r) for res in res_ci_pi])
    # ax_2.errorbar(x, y, yerr=e, fmt='o')
    # ax_2.set_xticklabels(param_name_list, rotation=-90)
# 
    # fig_3, ax_3 = plt.subplots(1, 1)
    # x = param_name_list
    # y = np.array([res.miu for res in res_ci_pi])
    # e = np.array([np.abs(res.ci_miu_l-res.ci_miu_r) for res in res_ci_pi])
    # ax_3.errorbar(x, y, yerr=e, fmt='o')
    # ax_3.set_xticklabels(param_name_list, rotation=-90)
    # ax_3.set_ylabel('slopes determined from linear regression')

    for param_name in param_name_list:
        data_temp = np.zeros([16, len(test.data_pool)])
        x = [i+1 for i in range(data_temp.shape[1])]
        id_list = []
        for num, sample in enumerate(test.data_pool):
            data_temp[:, num] = sample.roughness_param[param_name]
            id_list.append(sample.sample_name[:2])
        fig, ax = plt.subplots(1, 1)
        ax.boxplot(data_temp[:, :],
                   vert=True,  # vertical box alignment
                   patch_artist=False,  # fill with color
                   # notch=True,       # notch types
                   labels=id_list,
                   positions=x
                   )
        ax.set_ylabel(param_name, fontsize=13)
        ax.set_xlabel('Sample ID', fontsize=13)


'''


