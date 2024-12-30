import sys
import matplotlib

matplotlib.use('TkAgg')


sys.path.insert(1, '/Users/laptopd/Documents/Compositionality/Analysis/')

from analysis_scripts.behavior.behavior_scripts import *

if __name__=='__main__':

    sub = 'N'
    if sub == 'N':
        session_file = open('/Users/laptopd/Documents/Compositionality/Analysis/sessions_N.obj', 'rb')
        sessions = pickle.load(session_file)
        session_file.close()

        params={
            'subject': sub,
            'do_plot_mse_n_confusion': 1,
            'only_last_session': 1,
            'do_plot_each_confusion': 1,
            'is_invisible': 1
        }

        mse_n_confusion(dates.train_on_axis_dates_N, sessions, params)

