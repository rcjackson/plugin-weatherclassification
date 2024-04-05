from matplotlib import use 
use('Agg')
import numpy as np
import xarray as xr
import pandas as pd
import time
import argparse
import os
import xarray as xr
import tensorflow as tf
import glob
import matplotlib.pyplot as plt
import base64
import matplotlib.cm as cm

from waggle.plugin import Plugin
from datetime import datetime, timedelta
from scipy.signal import convolve2d
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input

from glob import glob
from datetime import datetime, timedelta
# 1. import standard logging module
import utils
import paramiko
import logging
logging.basicConfig(level=logging.DEBUG)


def convert_to_hours_minutes_seconds(decimal_hour, initial_time):
    delta = timedelta(hours=decimal_hour)
    return datetime(initial_time.year, initial_time.month, initial_time.day) + delta

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
    
    ds['snr'] = ds['snr'] + 2 * np.log10(ranges + 1)
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
        
        if not os.path.exists('/app/imgs/train'):
            os.mkdir('/app/imgs/train')

        fname = '/app/imgs/train/%d.png' % i
        width = first_shape[0]
        height = first_shape[1]
        # norm = norm.SerializeToStri
        fig, ax = plt.subplots(1, 1, figsize=(1 * (height / width), 1))
        # ax.imshow(my_data)
        ax.pcolormesh(my_data, cmap='HomeyerRainbow', vmin=20, vmax=150)
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
            sftp.get(file_name, os.path.join('/data', name))

lidar_ip_addr = os.environ["LIDAR_IP"]
lidar_uname = os.environ["LIDAR_USERNAME"]
lidar_pwd = base64.b64decode(os.environ["LIDAR_PASSWORD"]).decode("utf-8")

def worker_main(args):
    logging.debug("Loading model %s" % args.model)
    with open(args.model + ".json", 'r') as json_file:
        json_text = json_file.read()
    model = model_from_json(json_text)
    logging.debug("Model loaded")
    model.load_weights(args.model + "_weights.h5")
    logging.debug(args.model + "_weights.h5 loaded")
    interval = int(args.interval)
    logging.debug('opening input %s' % args.input)
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
    get_file(imgtime, lidar_ip_addr, lidar_uname, lidar_pwd)

    run = True
    already_done = []
    with Plugin() as plugin:
        while run:
            class_names = ['clear', 'cloudy']

            stare_list = glob(os.path.join(args.input, 'Stare*.hpl'))
            
            for fi in stare_list:
                logging.debug("Processing %s" % fi)
                dsd_ds = load_file(fi)
                logging.debug(dsd_ds)
                time_list = make_imgs(dsd_ds, args.config)
                dsd_ds.close()
                file_list = glob('/app/imgs/*.png')
                logging.debug(file_list)
                
                img_gen = ImageDataGenerator(
                    preprocessing_function=preprocess_input)

                gen = img_gen.flow_from_directory(
                         '/app/imgs/', target_size=(256, 128), shuffle=False)
                out_predict = model.predict(gen).argmax(axis=1)
                
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


def main(args):
    if args.verbose:
        logging.debug('running in a verbose mode')
    worker_main(args)


if __name__ == '__main__':
    # Register colrmap for images
    cm.register_cmap('HomeyerRainbow', yuv_rainbow_24(15))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--verbose', dest='verbose',
        action='store_true', help='Verbose')
    parser.add_argument(
        '--input', dest='input',
        action='store', default='/data',
        help='Path to input device or ARM datastream name')
    
    parser.add_argument(
        '--model', dest='model',
        action='store', default='resnet50',
        help='Path to model')
    parser.add_argument(
        '--interval', dest='interval',
        action='store', default=0,
        help='Time interval in seconds')
    parser.add_argument(
            '--loop', action='store_true')
    parser.add_argument('--no-loop', action='store_false')
    parser.set_defaults(loop=True)
    parser.add_argument(
            '--config', dest='config', action='store', default='dlacf',
            help='Set to User5 for PPI or Stare for VPTs')
    parser.add_argument('--date', dest='date', action='store',
                        default=None,
                        help='Date of record to pull in (YYYYMMDD)')
    parser.add_argument('--time', dest='time', action='store',
                        default=None, help='Time of record to pull (hour)')
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    try:
        main(parser.parse_args())
    except Exception as e:
        logging.debug(e)
                                            
