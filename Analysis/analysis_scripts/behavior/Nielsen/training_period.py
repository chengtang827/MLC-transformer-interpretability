import sys
import matplotlib

matplotlib.use('TkAgg')


sys.path.insert(1, '/Users/laptopd/Documents/Compositionality/Analysis/')

from analysis_scripts.behavior.behavior_scripts import *

if __name__=='__main__':

    sub = 'F'
    if sub == 'F':
        session_file = open('/Users/laptopd/Documents/Compositionality/Analysis/sessions_F.obj', 'rb')
        sessions = pickle.load(session_file)
        session_file.close()

        params={
            'subject': 'F',
            'do_plot_mse_n_confusion': 1,
            'do_plot_each_confusion': 1,
            'is_invisible': 1
        }
        mse_n_confusion(dates.train_dates_F, sessions, params)

