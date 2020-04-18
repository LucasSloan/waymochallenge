FROM tensorflow/tensorflow:2.2.0rc3-gpu
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git
RUN pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'