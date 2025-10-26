# Use a lightweight Miniconda image
FROM continuumio/miniconda3:latest

# Set working directory
WORKDIR /app

# Copy environment file first to leverage Docker caching
COPY environment.yml .

# Create the conda environment from the yml file
RUN conda env create -f environment.yml

# Make sure the environment is activated in every RUN/CMD
SHELL ["conda", "run", "-n", "etl_env", "/bin/bash", "-c"]

# Copy the rest of the app code
COPY . .

# Set default command to run your ETL script
CMD ["python", "etl.py"]