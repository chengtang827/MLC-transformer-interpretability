import sys
import matplotlib

matplotlib.use('TkAgg')


sys.path.insert(1, '/Users/laptopd/Documents/Compositionality/Analysis/')

from run_parse_trials import *
from analysis_scripts.behavior.behavior_scripts import *

sub = 'F'

if sub == 'F':
    session_file = open('/Users/laptopd/Documents/Compositionality/Analysis/sessions_F.obj', 'rb')
    sessions = pickle.load(session_file)
    session_file.close()

    # training sessions on the whole screen,
    session_dates = ['20230721', '20230724', '20230725', '20230726', '20230727', '20230728', '20230730', '20230731',
                     '20230801', '20230802', '20230803', '20230804', '20230807', '20230809', '20230811', '20230815',
                     '20230816', '20230818', '20230820', '20230821', '20230822', '20230823', '20230824', '20230825',
                     '20230827', '20230828', '20230829', '20230903', '20230904', '20230905', '20230906', '20230907',
                     '20230908', '20230911', '20230912', '20230913', '20230914', '20230916', '20230918', '20230919',
                     '20230920', '20230925']
    #
    params={
        'is_trained': 1,
        'is_invisible': 0
    }
    mse_n_confusion(session_dates, sessions, params)

