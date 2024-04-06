from matplotlib import use 
use('Agg')
import numpy as np
import xarray as xr
import pandas as pd
import time
import argparse
import tensorflow as tf
import os
import xarray as xr
import glob
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import base64
import matplotlib.cm as cm
import cv2

from waggle.plugin import Plugin
from datetime import datetime, timedelta
from scipy.signal import convolve2d
from glob import glob
from datetime import datetime, timedelta
from tensorflow.keras.applications.resnet import preprocess_input

# 1. import standard logging module
import utils
import paramiko
import logging
logging.basicConfig(level=logging.DEBUG)


def convert_to_hours_minutes_seconds(decimal_hour, initial_time):
    delta = timedelta(hours=decimal_hour)
    return datetime(initial_time.year, initial_time.month, initial_time.day) + delta

def _generate_cmap(name, spec, lutsize):
    """Generates the requested cmap from it's name *name*. The lut size is
    *lutsize*."""

    # Generate the colormap object.
    if isinstance(spec, dict) and 'red' in spec.keys():
        return colors.LinearSegmentedColormap(name, spec, lutsize)
    else:
        return colors.LinearSegmentedColormap.from_list(name, spec, lutsize)

def yuv_rainbow_24(nc):
    path1 = np.linspace(0.8 * np.pi, 1.8 * np.pi, nc)
    path2 = np.linspace(-0.33 * np.pi, 0.33 * np.pi, nc)

    y = np.concatenate(
        [np.linspace(0.3, 0.85, nc * 2 // 5), np.linspace(0.9, 0.0, nc - nc * 2 // 5)]
    )
    u = 0.40 * np.sin(path1)
    v = 0.55 * np.sin(path2) + 0.1

    rgb_from_yuv = np.array([[1, 0, 1.13983], [1, -0.39465, -0.58060], [1, 2.03211, 0]])
    cmap_dict = {'blue': [], 'green': [], 'red': []}
    for i in range(len(y)):
        yuv = np.array([y[i], u[i], v[i]])
        rgb = rgb_from_yuv.dot(yuv)
        red_tuple = (i / (len(y) - 1.0), rgb[0], rgb[0])
        green_tuple = (i / (len(y) - 1.0), rgb[1], rgb[1])
        blue_tuple = (i / (len(y) - 1.0), rgb[2], rgb[2])
        cmap_dict['blue'].append(blue_tuple)
        cmap_dict['red'].append(red_tuple)
        cmap_dict['green'].append(green_tuple)

    return cmap_dict


def load_file(file):
    field_dict = utils.hpl2dict(file)
    initial_time = pd.to_datetime(field_dict['start_time'])
    
    time = pd.to_datetime([convert_to_hours_minutes_seconds(x, initial_time) 
        for x in field_dict['decimal_time']])

    ds = xr.Dataset(coords={'range':field_dict['center_of_gates'],
                            'time': time,
                            'azimuth': ('time', field_dict['azimuth'])},
                    data_vars={'radial_velocity':(['range', 'time'],
                                                  field_dict['radial_velocity']),
                               'beta': (('range', 'time'), 
                                        field_dict['beta']),
                               'intensity': (('range', 'time'),
                                             field_dict['intensity'])
                              }
                   )
    ds['snr'] = ds['intensity'] - 1
    return ds


def return_convolution_matrix(time_window, range_window):
    return np.ones((time_window, range_window)) / (time_window * range_window)

def make_imgs(ds, config, interval=5):
    range_bins = np.arange(60., 11280., 120.)
    # Parse model string for locations of snr, mean_velocity, spectral_width
    locs = 0
    snr_thresholds = []
    scp_ds = {}
    interval = 5
    dates = pd.date_range(ds.time.values[0], ds.time.values[-1], freq='%dmin' % interval)
    
    times = ds.time.values
    logging.debug(times)
    which_ranges = int(np.argwhere(ds.range.values < 8000.)[-1])
    ranges = np.tile(ds.range.values, (ds['snr'].shape[1], 1)).T
    
    ds['snr'] = ds['intensity']
    conv_matrix = return_convolution_matrix(5, 5)
    snr_avg = convolve2d(ds['snr'].values, conv_matrix, mode='same')
    ds['stddev'] = (('range', 'time'), 
            np.sqrt(convolve2d((ds['snr'] - snr_avg) ** 2, conv_matrix, mode='same')))
    Zn = ds.stddev.values.T

    cur_time = times[0]
    end_time = times[-1]
    time_list = []
    start_ind = 0
    i = 0
    first_shape = None

    while cur_time < end_time:
        next_time = cur_time + np.timedelta64(interval, 'm')
        logging.debug((next_time, end_time))

        if next_time > end_time:
            next_ind = len(times)
        else:
            next_ind = np.argmin(np.abs(next_time - times))
        if (start_ind >= next_ind):
            break

        my_data = Zn[start_ind:next_ind, 0:which_ranges].T

        my_times = times[start_ind:next_ind]
        if len(my_times) == 0:
            break
        start_ind += next_ind - start_ind + 1

        if first_shape is None:
            first_shape = my_data.shape
        else:
            if my_data.shape[0] > first_shape[0]:
                my_data = my_data[:first_shape[0], :]
            elif my_data.shape[0] < first_shape[0]:
                my_data = np.pad(my_data, [(0, first_shape[0] - my_data.shape[0]), (0, 0)],
                                 mode='constant')
        if not os.path.exists('imgs'):
            os.mkdir('imgs')


        if not os.path.exists('/app/imgs/'):
            os.mkdir('/app/imgs')
        

        fname = '/app/imgs/%d.png' % i
        width = first_shape[0]
        height = first_shape[1]
        # norm = norm.SerializeToStri
        fig, ax = plt.subplots(1, 1, figsize=(1, 1 * (height/width)))
        # ax.imshow(my_data)
        ax.pcolormesh(my_data, cmap='HomeyerRainbow', vmin=0, vmax=5)
        ax.set_axis_off()
        ax.margins(0, 0)
        try:
            fig.savefig(fname, dpi=300, bbox_inches='tight', pad_inches=0)
        except RuntimeError:
            plt.close(fig)
            continue

        plt.close(fig)
        i = i + 1
        del fig, ax
        time_list.append(cur_time)
        cur_time = next_time

    return time_list


def progress(bytes_so_far: int, total_bytes: int):
    pct_complete = 100. * float(bytes_so_far) / float(total_bytes)
    if int(pct_complete * 10) % 100 == 0:
        logging.debug("Total progress = %4.2f" % pct_complete)

def get_file(time, lidar_ip_addr, lidar_uname, lidar_pwd):
    with paramiko.SSHClient() as ssh:
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        logging.debug("Connecting to %s" % lidar_ip_addr)
        ssh.connect(lidar_ip_addr, username=lidar_uname, password=lidar_pwd)
        logging.debug("Connected to the Lidar!")
        year = time.year
        day = time.day
        month = time.month
        hour = time.hour

        file_path = "/C:/Lidar/Data/Proc/%d/%d%02d/%d%02d%02d/" % (year, year, month, year, month, day)
        logging.debug(file_path)
        with ssh.open_sftp() as sftp:
            file_list = sftp.listdir(file_path)
            time_string = '%d%02d%02d_%02d' % (year, month, day, hour) 
            file_name = None
            
            for f in file_list:
                if time_string in f:
                    file_name = f
            if file_name is None:
                logging.debug("%s not found!" % str(time))
            base, name = os.path.split(file_name)
            logging.debug(print(file_name))
            sftp.get(os.path.join(file_path, file_name), name)


class TFLiteModel:
    def __init__(self, model_path: str):
        self.interpreter = tf.lite.Interpreter(model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def predict(self, *data_args):
        assert len(data_args) == len(self.input_details)
        for data, details in zip(data_args, self.input_details):
            self.interpreter.set_tensor(details["index"], data)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_details[0]["index"])

def worker_main(args, config_dict):
    logging.debug("Loading model %s" % args.model)
    interval = int(args.interval)
    if args.date is None and args.time is None:
        imgtime = datetime.now() - timedelta(hours=1)
    elif args.date is None:
        cur_day = datetime.now()
        imgtime = datetime(cur_day.year, cur_day.month, cur_day.day, str(args.time))
    else:
        year = int(args.date[0:4])
        month = int(args.date[4:6])
        day = int(args.date[6:8])
        hour = int(args.time)
        imgtime = datetime(year, month, day, hour)
    get_file(imgtime, args.IP, args.uname, args.password)

    run = True
    already_done = []
    with Plugin() as plugin:
        while run:
            class_names = ['clear', 'cloudy']

            stare_list = glob('Stare*.hpl')
            
            for fi in stare_list:
                logging.debug("Processing %s" % fi)
                dsd_ds = load_file(fi)
                logging.debug(dsd_ds)
                time_list = make_imgs(dsd_ds, args.config)
                dsd_ds.close()
                del dsd_ds
                model = TFLiteModel(args.model)
                file_list = glob('/app/imgs/*.png')
                logging.debug(file_list)
                out_predict = []
                for f in file_list:
                    image = cv2.imread(f)
                    image = cv2.resize(image, (128, 96))
                    image = image.astype(np.float32)[np.newaxis]
                    image = preprocess_input(image)
                    out_predict.append(model.predict(image)[0].argmax())

                for i, ti in enumerate(time_list):
                    if ti not in already_done:
                        tstamp = int(ti)
                        
                        if out_predict[i] == 0:
                            string = "clear"
                        else:
                            string = "clouds/rain"
                        logging.debug("%s: %s" % (str(ti), string))

                        plugin.publish("weather.classifier.class",
                                int(out_predict[i]),
                                timestamp=tstamp)
                        already_done.append(ti)

                for f in file_list:
                    plugin.upload_file(f)                   
            if args.loop == False:
                run = False


def main(args, config):
    if args.verbose:
        logging.debug('running in a verbose mode')
    worker_main(args, config)


if __name__ == '__main__':
    # Register colrmap for images
    default_config = {
        'num_epochs': 2000,
        'learning_rate': 0.001,
        'num_nodes': 512,
        'num_layers': 3,
        'batch_size': 32}

    LUTSIZE = mpl.rcParams['image.lut']
    cmap = _generate_cmap('HomeyerRainbow', yuv_rainbow_24(15), LUTSIZE)
    cm.register_cmap(cmap=cmap, name='HomeyerRainbow')
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--verbose', dest='verbose',
        action='store_true', help='Verbose')
    parser.add_argument(
            '--password', dest='password',
            default='', action='store', 
            help='Lidar password')
    parser.add_argument(
            '--IP', default='10.31.81.87',
            dest='IP', action='store',
            help='Lidar IP address')

    parser.add_argument(
            '--uname', default='end user',
            dest='uname', action='store',
            help='Lidar username')
    parser.add_argument(
        '--model', dest='model',
        action='store', default='/app/model.tflite',
        help='Path to model')
    parser.add_argument(
        '--interval', dest='interval',
        action='store', default=0,
        help='Time interval in seconds')
    parser.add_argument(
            '--loop', action='store_true')
    parser.add_argument('--no-loop', action='store_false')
    parser.set_defaults(loop=False)
    parser.add_argument(
            '--config', dest='config', action='store', default='dlacf',
            help='Set to User5 for PPI or Stare for VPTs')
    parser.add_argument('--date', dest='date', action='store',
                        default=None,
                        help='Date of record to pull in (YYYYMMDD)')
    parser.add_argument('--time', dest='time', action='store',
                        default=None, help='Time of record to pull (hour)')
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                    gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=512)])
        except RuntimeError as e:
            print(e)
    main(parser.parse_args(), default_config)
                                            
