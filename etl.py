import pandas as pd
from sqlalchemy import create_engine
import unicodedata
import re
import numpy as np
import xml.etree.ElementTree as ET
from datetime import datetime
import json

#import streamlit as st


# -------------------------
# 1 CONFIGURATION
# -------------------------
CSV_FILE_ENCOUNTERS = r"C:\Users\Patrick\Desktop\UK-assignment\encounters.csv"
CSV_FILE_PATIENTS = r"C:\Users\Patrick\Desktop\UK-assignment\patients.csv"  
XML_FILE_DIAGNOSES = r"C:\Users\Patrick\Desktop\UK-assignment\diagnoses.xml"  
POSTGRES_USER = "postgres"
POSTGRES_PASSWORD = "start"
POSTGRES_HOST = "localhost"
POSTGRES_PORT = 5432
POSTGRES_DB = "postgres"
TABLE_NAME_PATIENTS = "patients"
TABLE_NAME_ENCOUNTERS = "encounters"
TABLE_NAME_DIAGNOSES= "diagnoses"
# -------------------------
# 2️ EXTRACT: Load CSV
# -------------------------
# Use utf-8-sig to handle BOM if present
patients_df = pd.read_csv(CSV_FILE_PATIENTS, encoding='utf-8-sig')

# Method for cleaning up encounters.csv
def read_messy_encounters(filepath):
    """
    Read a mixed-delimiter encounters file without dropping any rows.
    Keeps both ',' and ';' lines, normalizes to consistent columns.
    """
    cleaned_lines = []
    expected_cols = [
        "encounter_id", "patient_id", "admit_dt",
        "discharge_dt", "encounter_type", "source_file"
    ]

    # Read raw text
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Skip duplicate header lines
            if line.lower().startswith("encounter_id"):
                continue

            # Split based on delimiter
            if ";" in line and "," not in line:
                parts = line.split(";")
            else:
                parts = [p.strip() for p in line.split(",")]

            # If there are too many fields, trim extras
            if len(parts) > len(expected_cols):
                parts = parts[:len(expected_cols)]

            # If too few, pad with Nones
            elif len(parts) < len(expected_cols):
                parts += [None] * (len(expected_cols) - len(parts))

            cleaned_lines.append(parts)

    # Create DataFrame manually
    df = pd.DataFrame(cleaned_lines, columns=expected_cols)

    # Clean up whitespace from all string cells
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    return df
#encounters_df = pd.read_csv("encounters.csv", sep=None, engine="python", on_bad_lines="skip")

encounters_df = read_messy_encounters(CSV_FILE_ENCOUNTERS)

print(encounters_df)



# Method for extracting Diagnoses from CSV

def parse_diagnoses_to_df(xml_file: str) -> pd.DataFrame:
    ns = {"ns": "http://example.org/diagnosis"}
    tree = ET.parse(xml_file)
    root = tree.getroot()

    records = []
    for diag in root.findall("ns:Diagnosis", ns):
        encounter_id = diag.findtext("ns:encounterId", default=None, namespaces=ns)
        code_elem = diag.find("ns:code", ns)
        code = code_elem.text if code_elem is not None else None
        code_system = code_elem.get("system") if code_elem is not None else None
        is_primary_text = diag.findtext("ns:isPrimary", default=None, namespaces=ns)
        is_primary = (
            is_primary_text.lower() == "true" if is_primary_text is not None else None
        )
        recorded_at = diag.findtext("ns:recordedAt", default=None, namespaces=ns)

        # Normalize datetime (handles missing timezone or date-only)
        if recorded_at:
            try:
                recorded_at = datetime.fromisoformat(recorded_at)
            except ValueError:
                recorded_at = datetime.fromisoformat(recorded_at + "T00:00:00")

        records.append(
            {
                "encounter_id": encounter_id,
                "code": code,
                "code_system": code_system,
                "is_primary": is_primary,
                "recorded_at": recorded_at,
            }
        )

    df = pd.DataFrame(records)
    return df

diagnoses_df = parse_diagnoses_to_df(XML_FILE_DIAGNOSES)
print(diagnoses_df)
# -------------------------
# 3️ TRANSFORM: Clean text
# -------------------------

##DATA NORMALIZATION
# Height Column needs to be normalized


#define methods

def height_to_cm(value):
    if pd.isna(value):
        return np.nan

    s = str(value).strip().lower()

    # Handle meters (e.g., 1.8m)
    if re.search(r'\b\d+(\.\d+)?\s*m\b', s) and 'cm' not in s:
        num = float(re.findall(r'[\d.]+', s)[0])
        return num * 100

    # Handle centimeters (e.g., 180cm)
    if 'cm' in s:
        num = float(re.findall(r'[\d.]+', s)[0])
        return num

    # Handle feet/inches combined (e.g., 5ft 6in, 5'6", 5'6)
    ft_in_pattern = re.match(r"(\d+)\s*(?:ft|')\s*(\d+)?\s*(?:in|\"|$)?", s)
    if ft_in_pattern:
        feet = int(ft_in_pattern.group(1))
        inches = int(ft_in_pattern.group(2)) if ft_in_pattern.group(2) else 0
        return feet * 30.48 + inches * 2.54

    # Handle inches only (e.g., 70 in)
    if 'in' in s:
        num = float(re.findall(r'[\d.]+', s)[0])
        return num * 2.54

    # Handle plain numbers (ambiguous)
    if re.fullmatch(r'[\d.]+', s):
        val = float(s)
        # assume numbers < 3 are in meters (e.g., 1.75)
        if val < 3:
            return val * 100
        # assume numbers > 100 are already cm
        return val

    return np.nan  # unrecognized






def clean_utf8(s):
    """Ensure string is valid UTF-8 and normalize unicode."""
    if isinstance(s, str):
        # Replace invalid bytes with replacement character
        s = s.encode('utf-8', 'replace').decode('utf-8')
        # Normalize unicode (NFKC form)
        s = unicodedata.normalize('NFKC', s)
        s = s.lower()
    return s

# Normalize Column Names and remove whitespaces

patients_df.columns = (
    patients_df.columns
    .str.strip()            # remove leading/trailing spaces
    .str.replace('\ufeff', '', regex=True)  # remove BOM
    .str.replace(' +', '_', regex=True)     # replace internal spaces with underscores
    .str.lower()            # make lowercase for consistency
)
encounters_df.columns = (
    encounters_df.columns
    .str.strip()            # remove leading/trailing spaces
    .str.replace('\ufeff', '', regex=True)  # remove BOM
    .str.replace(' +', '_', regex=True)     # replace internal spaces with underscores
    .str.lower()            # make lowercase for consistency
)




def weight_to_kg(value, height_cm=None):
    """
    Convert mixed-format weight values to kilograms.
    Uses height to infer units and fixes implausible BMI values.
    Returns: (weight_kg, flagged, original_val, reason)
    """
    if pd.isna(value):
        return np.nan, True, value, "missing_value"

    s = str(value).strip().lower()

    # Handle missing markers
    if s in {"", "na", "n/a", "none", "null", "-"}:
        return np.nan, True, value, "missing_marker"

    nums = re.findall(r"[\d.]+", s)
    if not nums:
        return np.nan, True, value, "no_numeric_found"

    val = float(nums[0])
    flagged = False
    reason = "ok"

    # --- Explicit units ---
    if "kg" in s:
        weight_kg = val
    elif "lb" in s:
        weight_kg = val * 0.453592
    else:
        # --- No units provided ---
        flagged = True
        reason = "missing_unit"

        if pd.notna(height_cm):
            bmi_kg = val / ((height_cm / 100) ** 2)
            bmi_lb = (val * 0.453592) / ((height_cm / 100) ** 2)
            if 10 <= bmi_kg <= 45:
                weight_kg = val
                reason = "missing_unit_assumed_kg"
            elif 10 <= bmi_lb <= 45:
                weight_kg = val * 0.453592
                reason = "missing_unit_assumed_lb"
            else:
                weight_kg = val * 0.453592 if val > 140 else val
                reason = "ambiguous_missing_unit"
        else:
            weight_kg = val * 0.453592 if val > 140 else val
            reason = "no_height_missing_unit"

    # --- Plausibility check ---
    if pd.notna(height_cm):
        bmi = weight_kg / ((height_cm / 100) ** 2)
        if bmi < 10 or bmi > 60:
            # try flipping units
            flipped = weight_kg / 0.453592 if "kg" in s else weight_kg * 0.453592
            new_bmi = flipped / ((height_cm / 100) ** 2)
            if 10 <= new_bmi <= 45:
                flagged = True
                reason = "implausible_bmi_fixed"
                weight_kg = flipped
            else:
                flagged = True
                reason = f"implausible_bmi_{bmi:.1f}"

    return weight_kg, flagged, value, reason


def normalize_weights(df, filename=CSV_FILE_PATIENTS):
    """
    Apply weight normalization to DataFrame.
    Returns: (cleaned_df, data_quality_log_df)
    """
    

    def process_row(row):
        w_kg, flagged, orig, reason = weight_to_kg(row["weight"], row.get("height_cm"))
        if flagged:
            logs.append({
                "patient_id": row.get("patient_id"),
                "filename": filename,
                "original_value": str(orig),
                "cleaned_value": str(w_kg),
                "column_name": "weight",
                "reason": reason
            })
        return w_kg

    df["weight_kg"] = df.apply(process_row, axis=1)
    return df



def map_sex_to_iso5218_with_log(df, filename=CSV_FILE_PATIENTS):
    """
    Map the 'sex' column to ISO/IEC 5218 numeric codes:
        M → 1, F → 2, U → 0, O → 9
    Any other or missing value → 0 (Unknown)
    
    Returns:
        df        - original DataFrame with new column 'sex_iso'
        log_df    - DataFrame with all rows that were mapped to 0
    """
    mapping = {
        "M": 1,  # Male
        "F": 2,  # Female
        "U": 0,  # Unknown
        "O": 9,  # Other / Not applicable
    }


    def map_value(row):
        value = row.get("sex")
        if pd.isna(value):
            logs.append({
                "patient_id": row.get("patient_id"),
                "filename": filename,
                "original_value": str(value),
                "mapped_code": "0",
                "reason": "missing_value"
            })
            return 0

        s = str(value).strip().upper()
        if s not in mapping:
            logs.append({
                "patient_id": row.get("patient_id"),
                "filename": filename,
                "original_value": str(value),
                "mapped_code": "0",
                "reason": "invalid_code"
            })
            return 0

        code = mapping[s]

        # log only if it's 0 (U or invalid/missing)
        if code == 0:
            logs.append({
                "patient_id": row.get("patient_id"),
                "filename": filename,
                "original_value": str(value),
                "cleaned_value": str(code),
                "reason": "unknown_or_missing"
            })
        return code

    df["sex_iso"] = df.apply(map_value, axis=1)
    return df


def parse_dob(df, dob_column="dob", filename="patients.csv"):
    """
    Parse a 'dob' column into datetime, assuming year-first formats (YYYY-MM-DD or YYYY/MM/DD).
    Handles some ambiguous cases by trying day-first if initial parsing fails.
    
    Returns:
        df       – original DataFrame with new column 'dob_parsed' (datetime)
        log_df   – DataFrame with rows that could not be parsed, with patient_id and filename
    """

    def parse_date(row):
        dob_val = row.get(dob_column)
        patient_id = row.get("patient_id")
        if pd.isna(dob_val):
            logs.append({
                "patient_id": patient_id,
                "filename": filename,
                "column_name": dob_column,
                "original_value": dob_val,
                "cleaned_value": None,
                "reason": "missing_value"
            })
            return pd.NaT

        # First attempt: dayfirst=False
        try:
            parsed = pd.to_datetime(dob_val, errors="raise", dayfirst=False)
            return parsed
        except:
            # Second attempt: dayfirst=True
            try:
                parsed = pd.to_datetime(dob_val, errors="raise", dayfirst=True)
                logs.append({
                    "patient_id": patient_id,
                    "filename": filename,
                    "column_name": dob_column,
                    "original_value": dob_val,
                    "cleaned_value": parsed,
                    "reason": "day_month_swapped"
                })
                return parsed
            except:
                # Unparseable
                logs.append({
                    "patient_id": patient_id,
                    "filename": filename,
                    "column_name": dob_column,
                    "original_value": dob_val,
                    "cleaned_value": None,
                    "reason": "unparseable_date"
                })
                return pd.NaT

    df["dob_parsed"] = df.apply(parse_date, axis=1)
    return df





def remove_patient_duplicates(df):
    """
    Remove duplicate patient entries based on either:
      1. Duplicate patient_id, or
      2. Same name + dob + rounded height + rounded weight

    Returns:
        df_cleaned – DataFrame with duplicates removed
        log_df     – DataFrame of deleted duplicates
    """


    # --- Step 1: Ensure height/weight are numeric for rounding ---
    df["height_cm"] = pd.to_numeric(df["height_cm"], errors="coerce")
    df["weight_kg"] = pd.to_numeric(df["weight_kg"], errors="coerce")

    # --- Step 2: Create rounded helper columns for deduplication ---
    df["_height_rounded"] = df["height_cm"].round(0)
    df["_weight_rounded"] = df["weight_kg"].round(0)

    # --- Step 3: Mark duplicates by patient_id ---
    dup_id_mask = df.duplicated(subset=["patient_id"], keep="first")

    # --- Step 4: Mark duplicates by personal info ---
    dup_person_mask = df.duplicated(
        subset=["given_name", "family_name", "dob", "_height_rounded", "_weight_rounded"],
        keep="first"
    )

    # --- Step 5: Combine both duplicate sets ---
    dup_mask = dup_id_mask | dup_person_mask
    duplicates = df[dup_mask]

    # --- Step 6: Log duplicates before dropping ---
    for _, row in duplicates.iterrows():
        logs.append({
            "patient_id": row.get("patient_id"),
            "filename": CSV_FILE_PATIENTS,
            "column_name": "duplicate_entry",
            "original_value": str({
                "given_name": row.get("given_name"),
                "family_name": row.get("family_name"),
                "dob": str(row.get("dob")),
                "height_cm": row.get("height_cm"),
                "weight_cm": row.get("weight_cm")
            }),
            "cleaned_value": None,
            "reason": "duplicate_removed"
        })

    

    # --- Step 7: Drop duplicates and cleanup helper columns ---
    df_cleaned = df.drop_duplicates(
        subset=["patient_id"], keep="first"
    ).drop_duplicates(
        subset=["given_name", "family_name", "dob", "_height_rounded", "_weight_rounded"],
        keep="first"
    ).drop(columns=["_height_rounded", "_weight_rounded"])

    return df_cleaned

# Method for Encounters (Encapsuled)
def clean_encounters(df):
    """
    Clean and normalize encounter data.
    Produces a cleaned DataFrame and a detailed log DataFrame.
    """

    # --- Clean column names ---
    df.columns = (
        df.columns
        .str.strip()
        .str.replace("\ufeff", "", regex=True)
        .str.replace(" +", "_", regex=True)
        .str.lower()
    )

    # --- Clean text fields ---
    df = df.applymap(lambda x: clean_utf8(x) if isinstance(x, str) else x)

    # --- Parse datetimes robustly ---
    def parse_datetime_safe(s, colname, patient_id):
        if pd.isna(s) or str(s).strip() == "":
            return pd.NaT
        try:
            return pd.to_datetime(s, utc=True, errors="raise", dayfirst=False)
        except Exception:
            try:
                return pd.to_datetime(s, utc=True, errors="raise", dayfirst=True)
            except Exception:
                logs.append({
                    "patient_id": patient_id,
                    "filename": CSV_FILE_ENCOUNTERS,
                    "column_name": colname,
                    "original_value": s,
                    "cleaned_value": None,
                    "reason": "invalid_datetime_format"
                })
                return pd.NaT

    df["admit_dt"] = df.apply(
        lambda row: parse_datetime_safe(row["admit_dt"], "admit_dt", row.get("patient_id")), axis=1
    )
    df["discharge_dt"] = df.apply(
        lambda row: parse_datetime_safe(row["discharge_dt"], "discharge_dt", row.get("patient_id")), axis=1
    )

    # --- Remove duplicate encounter_id values ---
    dup_mask = df.duplicated(subset=["encounter_id"], keep=False)
    for _, row in df[dup_mask].iterrows():
        logs.append({
            "patient_id": row.get("patient_id"),
            "filename": row.get("source_file", CSV_FILE_ENCOUNTERS),
            "column_name": "encounter_id",
            "original_value": row.get("encounter_id"),
            "cleaned_value": None,
            "reason": "duplicate_encounter_id"
        })
    df = df.drop_duplicates(subset=["encounter_id"], keep="first")

    # --- Logical check: discharge before admit ---
    invalid_time_mask = (df["discharge_dt"] < df["admit_dt"])
    for _, row in df[invalid_time_mask].iterrows():
        logs.append({
            "patient_id": row.get("patient_id"),
            "filename": row.get("source_file", CSV_FILE_ENCOUNTERS),
            "column_name": "discharge_dt",
            "original_value": str(row.get("discharge_dt")),
            "cleaned_value": None,
            "reason": "discharge_before_admit"
        })
    df.loc[invalid_time_mask, "discharge_dt"] = pd.NaT

    # --- Missing discharge ---
    missing_discharge_mask = df["discharge_dt"].isna()
    for _, row in df[missing_discharge_mask].iterrows():
        logs.append({
            "patient_id": row.get("patient_id"),
            "filename": row.get("source_file", CSV_FILE_ENCOUNTERS),
            "column_name": "discharge_dt",
            "original_value": None,
            "cleaned_value": None,
            "reason": "missing_discharge"
        })

    # --- Invalid encounter_type ---
    valid_types = ["INPATIENT", "OUTPATIENT", "ED"]
    for _, row in df[~df["encounter_type"].isin(valid_types)].iterrows():
        logs.append({
            "patient_id": row.get("patient_id"),
            "filename": row.get("source_file", CSV_FILE_ENCOUNTERS),
            "column_name": "encounter_type",
            "original_value": row.get("encounter_type"),
            "cleaned_value": None,
            "reason": "invalid_encounter_type"
        })
    df.loc[~df["encounter_type"].isin(valid_types), "encounter_type"] = "UNKNOWN"

    # --- Optional: calculate length_of_stay (in hours) ---
    df["length_of_stay_hours"] = (
        (df["discharge_dt"] - df["admit_dt"])
        .dt.total_seconds() / 3600
    )

    for col in ["admit_dt", "discharge_dt"]:
        if col in df.columns:
            try:
                df[col] = df[col].dt.tz_localize(None)
            except AttributeError:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                df[col] = df[col].dt.tz_localize(None)

        # --- Return cleaned data + logs ---
    return df


def clean_diagnoses(df):

  # --- Clean + Log ---
    cleaned_rows = []
    seen_keys = set()

    for _, row in df.iterrows():
        original = row.to_dict()
        reason = None
        cleaned = row.copy()

        # Rule 1: Drop missing code
        if pd.isna(cleaned["code"]):
            reason = "dropped for missing code"
            logs.append(
                {
                    "encounter_id": original.get("encounter_id"),
                    "code": original.get("code"),
                    "reason": reason,
                    "original_value": json.dumps(original, default=str),
                    "cleaned_value": None,
                }
            )
            continue

        # Rule 2: Fill missing encounter_id
        if pd.isna(cleaned["encounter_id"]):
            cleaned["encounter_id"] = "UNKNOWN"
            reason =  "missing encounter_id"

        # Rule 3: Fill missing is_primary
        elif pd.isna(cleaned["is_primary"]):
            cleaned["is_primary"] = False
            reason = "filled missing is_primary"


        # Rule 4: Deduplicate by encounter_id + code
        key = (cleaned["encounter_id"], cleaned["code"])
        if key in seen_keys:
            reason = "duplicate encounter_id + code"
            logs.append(
                {
                    "encounter_id": cleaned["encounter_id"],
                    "code": cleaned["code"],
                    "reason": reason,
                    "original_value": json.dumps(original, default=str),
                    "cleaned_value": None,
                }
            )
            continue
        else:
            seen_keys.add(key)
        cleaned_rows.append(cleaned)

    df_clean = pd.DataFrame(cleaned_rows)
    return df_clean
# Execution of Transformation for Patients
logs = []
print(patients_df.head())
#TODO fix this so it gets logged properly too!!!
patients_df['height_cm'] = patients_df['height'].apply(height_to_cm)
#here the weight is applied row by row and the height in cm is used to guess the more likely metric (bmi between 10 and 45 is seen as plausible)
patients_df = normalize_weights(patients_df)
patients_df = map_sex_to_iso5218_with_log(patients_df)
patients_df = parse_dob(patients_df)
patients_df = remove_patient_duplicates(patients_df)
#drop unneeded columns
patients_df = patients_df.drop(columns=['weight', 'height'])

patients_df = patients_df.applymap(clean_utf8)
print("patients DF 559")
print(patients_df.head)


# Execution of Transformation for Encounters 

encounters_df = clean_encounters(encounters_df)
encounters_df = encounters_df.applymap(clean_utf8)
print(encounters_df)
  

# Execution of Transformation for Diagnoses
diagnoses_df = clean_diagnoses(diagnoses_df)
diagnoses_df = diagnoses_df.applymap(clean_utf8)


#Turn logs into DataFrame
log_df = pd.DataFrame(logs)
log_df = log_df.applymap(clean_utf8)  











# -------------------------
# 4. LOAD: Write to PostgreSQL
# -------------------------
# Create SQLAlchemy engine
engine = create_engine(
    f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)

# Load DataFrame into PostgreSQL
patients_df.to_sql(TABLE_NAME_PATIENTS, engine, if_exists='replace', index=False)
#encounters_df.to_sql(TABLE_NAME_ENCOUNTERS, engine, if_exists='replace', index=False)
diagnoses_df.to_sql(TABLE_NAME_DIAGNOSES, engine, if_exists='replace', index=False)
log_df.to_sql("logs", engine, if_exists='replace', index=False)
diagnoses_df.to_sql("diagnoses", engine, if_exists='replace', index=False)


print(f"✅ CSV loaded successfully to PostgreSQL table")














