# Use a lightweight Miniconda image
FROM continuumio/miniconda3:latest

# Set working directory
WORKDIR /app

# Copy environment file first to leverage Docker caching
COPY environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml && conda clean -a
# Create the conda environment from the yml file
SHELL ["conda", "run", "-n", "UK-ETL", "/bin/bash", "-c"]


# Make sure the environment is activated in every RUN/CMD
RUN conda env list

# Copy the rest of the app code
COPY etl.py /app/
COPY data/ /app/data/
COPY interactive_dashboard.py /app/
COPY environment.yml /tmp/environment.yml
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "UK-ETL"]
CMD ["bash", "-c", "python etl.py; streamlit run interactive_dashboard.py --server.port 8501 --server.headless true --server.address 0.0.0.0"]
