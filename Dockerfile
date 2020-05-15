FROM tensorflow/tensorflow:nightly-gpu-py3
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git
RUN pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
RUN pip install PyYAML