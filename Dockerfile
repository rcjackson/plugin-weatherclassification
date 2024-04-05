FROM waggle/plugin-base:1.1.1-ml-cuda10.2-l4t

RUN apt-get update -y
RUN apt-get install -y python3-tk
#RUN apt-get install -y python3-scipy
#RUN apt-get install -y libhdf5-serial-dev
RUN apt install -y gcc g++ gfortran libopenblas-dev liblapack-dev pkg-config python3-pip python3-dev

#RUN pip3 uninstall -y tensorflow
RUN pip3 install netcdf4
RUN pip3 install xarray
RUN pip3 install --upgrade pywaggle
RUN pip3 install paramiko
ENV MPLBACKEND="agg"
ENV LIDAR_IP=LIDAR_IP
ENV LIDAR_USERNAME=$LIDAR_USERNAME
ENV LIDAR_PASSWORD=$LIDAR_PASSWORD
COPY app/ /app/
COPY *.json /app/
COPY *.h5 /app/
COPY *.home_point /app/
COPY data /data/
COPY data/* /data/
WORKDIR /app

ENTRYPOINT ["python3", "/app/app.py"]

