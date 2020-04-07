ARG TIDYVERSE_TAG

FROM rocker/tidyverse:$TIDYVERSE_TAG

RUN apt-get update && apt-get install -y libpython3-dev python3-venv python3-pip libjpeg-dev

RUN installGithub.r rstudio/keras@09de409ece8b9a47a4f915ee7397c486fc1e9e91 

ENV PYTHON_PACKAGES="\
    numpy \
    pandas \
    tensorflow==2.0 \
    pillow \
" 

RUN pip3 install --upgrade pip && pip3 install --no-cache-dir $PYTHON_PACKAGES
