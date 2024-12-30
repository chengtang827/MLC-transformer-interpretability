import sys
import matplotlib
import pandas

matplotlib.use('TkAgg')
sys.path.insert(1, '/Users/laptopd/Documents/Compositionality/Analysis/')
from analysis_scripts.behavior.behavior_scripts import *
import logging
import json

KEY_MAP = {
    "target_xy": "target_xy",
    "ball_alpha": "ball_alpha",
    "behavior": "behavior",
    "reported_xy": "reported_xy",
    "start_time": "cue1_start_time",
    "pre_target_delay": "pre_target_delay",
    "saccade_onset_time": "saccade_onset",
    "end_time": "end_time",
    # "eye_xy": "eye_xy"
}
logging.basicConfig(level=logging.DEBUG)


OVER_WRITE = 1
if __name__=='__main__':

    sub = 'F'
    sessions = compile_sessions(session_start='20230105', session_end='20230517', subject=sub)
    save_path = Path('/Users/laptopd/Documents/Compositionality/Analysis/share/sessions')
    for session in sessions:
        date = session.date
        save_name = save_path / f"{date}.json"

        if os.path.exists(save_name) and not OVER_WRITE:
            logging.debug(f'{date} already exists, skip')

        session_dict = {
            k: eval(f"session.{v}", {'session': session})
            for k, v in KEY_MAP.items()
        }

        # if has toss, use modify invisible trials
        if not (session.toss_probe==-1).all():
            idx_invis_by_toss = np.where(session.toss_probe <= session.ratio_probe)[0]
            session_dict['ball_alpha'][idx_invis_by_toss] = 0

        # change shape for dataframe
        for k in [
            "target_xy",
            "reported_xy",
        ]:
            session_dict[k] = [session_dict[k][i, :] for i in range(session_dict[k].shape[0])]

        df = pandas.DataFrame(session_dict)
        df.to_json(save_name, orient='records', lines=True)

        logging.debug(f"{date} completed")



