import cv2
import matplotlib.pyplot as plt
import numpy as np
import pickle
from run_parse_trials import *
from mworks.data import MWKFile

def _get_session(sessions, date):
    """
    :param sessions:
    :param date:
    :return:
    """
    for idx in range(len(sessions)):
        if sessions[idx].date == date:
            return sessions[idx]
    return

def _extract_eye(session, start, end):
    '''
    :param session:
    :param start: begin trial index
    :param end: end trial index
    :return:
    '''

    dir_raw = '/Users/laptopd/Documents/Compositionality/Analysis/trials/'
    session_date = session.date
    filenames = os.listdir(dir_raw)
    for i in range(len(filenames)):
        if session_date not in filenames[i].split('-'):
            continue
        filename = filenames[i]

    f = MWKFile(dir_raw + filename)
    try:
        f.open()
        total = f.get_events(codes=['total'])
    except:
        return

    start_time = int(session.cue1_start_time[start]*1e6)
    end_time = int(session.end_time[end]*1e6)
    eye_x_event = f.get_events(codes=['eye_x'], time_range=[start_time, end_time])
    eye_y_event = f.get_events(codes=['eye_y'], time_range=[start_time, end_time])
    time_span = (end_time-start_time)/1e6

    eye_x = []
    eye_y = []
    [eye_x.append(i0.data) for i0 in eye_x_event]
    [eye_y.append(i0.data) for i0 in eye_y_event]
    eye_xy = np.stack((np.array(eye_x), np.array(eye_y))).T
    f.close()
    return eye_xy, time_span

def _print_trial_info(session, start, end):
    speed = 3

    target_set = session.target_set
    trial_idx = np.arange(start, end+1).astype(int).tolist()
    target_list = []
    [target_list.append(target_set[session.target_id[i]]) for i in trial_idx]
    is_correct_list = []
    [is_correct_list.append(int(i in session.true_correct.tolist())) for i in trial_idx]

    print('var target_list = '+ str(target_list))
    print('var is_correct_list = '+ str(is_correct_list))
    show_start_list = [0]
    for i in range(1, len(trial_idx)):
        idx = trial_idx[i]
        show_start_list.append(round(session.cue1_start_time[idx]-session.end_time[idx-1]-3.5, 4))
    print('var show_start_list = '+str(show_start_list))
    show_target_list = []

    for idx in trial_idx:
        cue1_time = abs(np.array(session.target_set[session.target_id[idx]][1]).sum()) / speed
        show_target_list.append(round(session.end_time[idx] - session.cue1_start_time[idx] -cue1_time, 4))
    print('var show_target_list = ' + str(show_target_list))
    return

def _load_video(videoname):
    video_capture = cv2.VideoCapture(videoname)
    frames = []
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break  # Break the loop when the video ends

        frames.append(frame)

    video_matrix = np.array(frames)
    video_capture.release()
    return video_matrix

sample_frequency = 1000  # datapoints/second
session_file = open('/Users/laptopd/Documents/Compositionality/Analysis/sessions.obj', 'rb')
sessions = pickle.load(session_file)
session_file.close()

session_date = '20231006'
session = _get_session(sessions, session_date)

eye_len = []
[eye_len.append(i.shape[0]) for i in session.eye_xy]
eye_len = np.array(eye_len)
session.trial_len*1000/eye_len
start_trial = 35
end_trial = 37
# todo find good trials
eye_xy, time_span = _extract_eye(session, start_trial, end_trial)

_print_trial_info(session, start_trial, end_trial)

## load the reconstructed trial
## need to customize this part for each trial
video_matrix = _load_video(videoname='/Users/laptopd/Documents/Compositionality/Analysis/analysis_scripts/behavior'
                                     '/video/recon.mov')
# should be 480x480 screen
width_screen, height_screen = video_matrix.shape[1], video_matrix.shape[2]
n_frames = video_matrix.shape[0]
fps = int(n_frames/time_span)

# width and height of the grid in the video
width_draw = 430
height_draw = 430
offset_x = 0
offset_y = 20  # pos=down, neg=up
##
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for saving video
out = cv2.VideoWriter('recon_eye.mp4', fourcc, int(fps*0.75), (width_screen, height_screen))

# Create an array of indices for the downsampled points
indices = np.arange(0, len(eye_xy), len(eye_xy) / n_frames).astype(int)

# Interpolate the downsampled points using NumPy's interp function
eye_x_transformed = np.interp(indices, np.arange(eye_xy.shape[0]), eye_xy[:, 0])
eye_y_transformed = np.interp(indices, np.arange(eye_xy.shape[0]), eye_xy[:, 1])

eye_x_transformed = (eye_x_transformed / 12 * width_draw / 2 + width_draw / 2).astype(int)
eye_y_transformed = (-eye_y_transformed / 12 * width_draw / 2 + width_draw / 2).astype(int)

eye_x_transformed += offset_x
eye_y_transformed += offset_y

eye_x_transformed[eye_x_transformed<0]=0
eye_x_transformed[eye_x_transformed >= width_screen]= width_screen - 1

eye_y_transformed[eye_y_transformed<0]=0
eye_y_transformed[eye_y_transformed >= height_screen]= height_screen - 1

eye_xy_transformed = np.stack((eye_x_transformed, eye_y_transformed)).T

# fig, ax = plt.subplots()
# ax.scatter(eye_xy_transformed[:,0], eye_xy_transformed[:,1])
# plt.show()

# Loop through the time dimension
for i in range(eye_xy_transformed.shape[0]):
    # Create a blank canvas
    frame = video_matrix[i, :, :, :]

    # Scale and draw the coordinates
    x, y = eye_xy_transformed[i, :]

    cv2.circle(frame, (x, y), 5, (200, 100, 100), -1)

    # Write the frame to the video
    out.write(frame)

# Release the video writer and close the output file
out.release()
cv2.destroyAllWindows()
