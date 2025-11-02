## Readme

### How to setup:
Clone this repository:

```bash
git clone https://github.com//patrickthomasius/Assignment-ETL.git
cd Assignment-ETL
```
### Setup this project with Docker (recommended)
to replicate the runtime, a Dockerfile and a docker-compose file were included in this project.

To build and run the container open a shell inside the folder and run

`docker build .` to setup the docker image

afterwards run

`docker-compose up -d .` to run the container. This will start a postgres service that runs a fresh postgres instance, as well as the ETL.py to populate the database and the streamlit application to visualize the database

To view the database in a browser on the system, connect to browser to `http://localhost:8501`
#### Ports
This project uses the following ports:

- **5432** – PostgreSQL database
- **8501** – Streamlit web app interface

Make sure these ports are available on your machine before running the project.

You can change them in the `docker-compose.yml` or run command and in the respective scripts (at the start of interactive_dashboard.py, and at the start of etl.py) if needed. 

### Setup this project (without Docker)

1. Create Conda Environment:
`conda env create -f environment.yml`
2. Run postgress locally with fitting credentials, or adjust credential in the rag_postgres_llm.py script:

`POSTGRES_USER=postgres
POSTGRES_PASSWORD=start
POSTGRES_DB=postgres
POSTGRES_PORT=5432`

3. Run the etl:
`python etl.py`

4. to load the visualization, run the visualization script
`streamlit run interactive_dashboard.py`
### Explanation of my approach:

Here, a ETL was setup that extracts, transforms, and loads Data from 3 possible sources, one containing Patient Data, one containing Encounters and one containing diagnoses. Each of these required specific transformations and is then loaded into separate tables.

 To start with the Patient Data, it is worth noting that several columns have inconsistent units of measure or data formats. 
Dates have different formatting types. To solve this the date format was normalized. The solution was done under the assumption that all dates have Months before Days. However, if future Data contains any twisted date formats (detectable by inconsistent values >12 for MM) there should be another validation check added.
Another Column where this was an issue are the Weight and Height Columns. Here, Units of measure were converted to cm and missing units of measure were assigned a probable unit. 
	For Weight, a similar approach was taken but height was taken into consideration. If a calculated BMI was highly improbable, even set units of measurements were overridden with plausible ones. 
Furthermore, Gender/Sex was reformatted to fit numerical connotation and duplicates were dropped.
For Transformations log entries were added in a single log table that is shared with the other datasources/Tables, which logs original and cleaned value, or just the original value for dropped rows. Rows were dropped on either the Patient_ID being identical or all other personal information being identical in case of wrong or mismatched Patient_ID.

For the encounters, similar approaches were taken for the transformation of date columns. Furthermore, the length of the stay for each encounter was calculated and added in hours. 

Diagnoses were loaded from the xml that requires a different parsing method but other than that no transformation were applied. In the future it might be beneficial to check the diagnosis codes against their respective code systems for validation. 

It was generally made sure that missing values dont cause errors, but are logged into the log table.

In the second “interactive_dashboard.py” script, a dashboard was created with streamlit that allows interactive display of data, box plots and scatter plots of two numerical values can be displayed. 

AI was used as coding assistance in this Project. 
