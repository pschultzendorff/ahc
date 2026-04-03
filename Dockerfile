FROM python:3.12.11-slim
LABEL title="ahc"
LABEL description="Adaptive homotopy continuation solver for incompressible two-phase flow in porous media"
LABEL version="0.1"
LABEL maintainer="Peter von Schultzendorff"
LABEL email="peter.schultzendorff@uib.no"

WORKDIR /app

# Install git (not included in python:3.12-slim)
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Install PorePy fork
RUN git clone https://github.com/pschultzendorff/porepy_hc \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e /app/porepy_hc[development,testing]

# Copy files into container and install
COPY . ./ahc
RUN pip install --no-cache-dir -e /app/ahc

# Matplotlib backend for .png files
ENV MPLBACKEND=Agg

CMD ["/app/ahc/run_all.sh"]