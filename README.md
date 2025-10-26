## Readme

### How to setup:
conda env create -f environment.yml

Start Postgresql/Pgadmin on your system.
Optionally, adjust parameters at the top of the etl.py script.
Run the etl.py script. 

Alternatively, the docker-compose.yml can be used to setup a docker image. 

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

In the second “interactive_dashboard.py” script, a dashboard was created with streamlit that allows interactive display of data, box plots and scatter plots of two numerical values can be displayed. 

AI was used as coding assistance in this Project. 
