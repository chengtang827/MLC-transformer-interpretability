import sys
import matplotlib

matplotlib.use('TkAgg')


sys.path.insert(1, '/Users/laptopd/Documents/Compositionality/Analysis/')

from analysis_scripts.behavior.behavior_scripts import *

if __name__=='__main__':

    sub = 'F'
    if sub == 'F':

        #
        params={
            'subject': sub,
            'do_plot_mse_n_confusion': 1,
            'only_last_session': 0,
            'do_plot_each_confusion': 1,
            'is_invisible': 1
        }
        mse_n_confusion(dates.generalize_dates_F, params)


