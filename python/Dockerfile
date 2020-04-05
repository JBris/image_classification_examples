ARG BASE_IMAGE

FROM $BASE_IMAGE

RUN apt-get update && apt-get install -y libgtk2.0-dev

ENV PYTHON_PACKAGES="\
    numpy \
    matplotlib \
    scipy \
    scikit-learn \
    pandas \
    nltk \
    seaborn \
    keras \
    tensorflow \
    opencv-contrib-python \
    imutils \
    tqdm \
" 

RUN pip install --upgrade pip && pip install --no-cache-dir $PYTHON_PACKAGES
