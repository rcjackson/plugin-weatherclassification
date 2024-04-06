FROM waggle/plugin-base:1.1.1-ml-cuda10.2-l4t

RUN apt-get update -y
RUN apt-get install -y libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran
RUN pip3 install -U pip testresources setuptools==49.6.0 
RUN pip3 install -U numpy==1.19.4 future==0.18.2 mock==3.0.5 h5py==2.10.0 keras_preprocessing==1.1.1 keras_applications==1.0.8 gast==0.2.2 futures protobuf pybind11

RUN pip3 install netcdf4
RUN pip3 install matplotlib
RUN pip3 install xarray
RUN pip3 install --upgrade pywaggle
RUN pip3 install paramiko
RUN pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v44 tensorflow==2.3.0+nv20.9
ENV MPLBACKEND="agg"
ENV LIDAR_PASSWORD=$LIDAR_PASSWORD
COPY app/ /app/
COPY *.tflite /app/
COPY *.home_point /app/
COPY data /data/
COPY data/* /data/
WORKDIR /app

ENTRYPOINT ["python3", "/app/app.py"]

