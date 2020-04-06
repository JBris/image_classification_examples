ARG TIDYVERSE_TAG

FROM rocker/tidyverse:$TIDYVERSE_TAG

RUN install2.r --error \
    --deps TRUE \     
    reticulate 

RUN installGithub.r rstudio/keras@09de409ece8b9a47a4f915ee7397c486fc1e9e91 

RUN Rscript -e 'reticulate::install_miniconda(path = "/opt/miniconda")'

RUN chmod +x -R /opt/miniconda
