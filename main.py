import os
import re
import unicodedata
import pandas as pd
from datetime import timedelta
import matplotlib.pyplot as plt

""" Load data """
BASE_DIR = os.path.dirname(__file__)
csv_path = os.path.join(BASE_DIR, "dataset", "studenti.csv")
df = pd.read_csv(csv_path)


""" Basic Analyze of datas """

print("First 10 rows: ")
print(df.head(10))

print("\nDataset info:")
print(df.info())

print("\nStatistical description of dataset:")
print(df.describe(include="all"))

# Num of rows and cols
rows, cols = df.shape
print(f"\nDataset has {rows} rows and {cols} columns")

print("\nColumns with NaN values:")
print(df.isna().sum())

""" Cleaning data """

def strip_accents(s: str) -> str:
  if not isinstance(s, str):
    return s
  nfkd = unicodedata.normalize("NFKD", s)
  return "".join(ch for ch in nfkd if not unicodedata.combining(ch))

def standardize_city(raw: str) -> str:
  if not isinstance(raw, str) or not raw.strip():
    return "Nepoznat"
  s = strip_accents(raw).strip().lower()
  mapping = {
    "nis": "Niš",
    "niš": "Niš",
    "beograd": "Beograd",
    "novi sad": "Novi Sad",
    "pg": "Podgorica",
    "podgorica": "Podgorica"
  }
  s = re.sub(r"\s+", " ", s)
  return mapping.get(s, raw.strip().title())

def clean_phone(raw: str) -> str | float:
  if not isinstance(raw, str):
    return float("nan")
  digits = re.sub(r"\D", "", raw)
  if not digits:
    return float("nan")
  digits = digits.lstrip("0")
  if digits.startswith("381"):
    rest = digits[3:]
  else:
    rest = digits
  if len(rest) < 8:
    return float("nan")
  if len(rest) > 10:
    rest = rest[-9:] if len(rest) >= 9 else rest[-10:]
  if len(rest) not in (9, 10):
    rest = rest[-9:]
    if len(rest) != 9:
      return float("nan")
  return f"+381{rest}"

def valid_email(s: str) -> bool:
  if not isinstance(s, str):
    return False
  s = s.strip()
  return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", s))

def coerce_date(s: str) -> pd.Timestamp | None:
  return pd.to_datetime(s, errors="coerce", dayfirst=False, infer_datetime_format=True)

age_col = "Godine"
if age_col in df.columns:
  df.loc[~df[age_col].between(16, 90), age_col] = pd.NA
  age_mean = df[age_col].astype("float").mean(skipna=True)
  df[age_col] = df[age_col].fillna(age_mean)

gpa_col = "Prosek"
if gpa_col in df.columns:
  df[gpa_col] = pd.to_numeric(df[gpa_col], errors="coerce")
  df.loc[~df[gpa_col].between(6.0, 10.0), gpa_col] = pd.NA
  gpa_median = df[gpa_col].median(skipna=True)
  df[gpa_col] = df[gpa_col].fillna(gpa_median).clip(lower=6.0, upper=10.0)

city_col = "Grad"
if city_col in df.columns:
  df[city_col] = df[city_col].fillna("nepoznat")

if city_col in df.columns:
  df[city_col] = df[city_col].apply(standardize_city)

if "ESPB" in df.columns:
  df["ESPB"] = pd.to_numeric(df["ESPB"], errors="coerce")
  df.loc[~df["ESPB"].between(0, 300), "ESPB"] = pd.NA
  df["ESPB"] = df["ESPB"].fillna(df["ESPB"].median(skipna=True))

if "Ime" in df.columns:
  df["Ime"] = df["Ime"].apply(lambda x: x.strip() if isinstance(x, str) else x)
  df.loc[df["Ime"].astype(str).str.len() == 0, "Ime"] = pd.NA

if "ID" in df.columns:
  df["ID"] = pd.to_numeric(df["ID"], errors="coerce").astype("Int64")
  df = df[df["ID"].notna() & (df["ID"] > 0)]

if "Email" in df.columns:
  df["Email"] = df["Email"].astype(str).str.strip()
  df = df[df["Email"].apply(valid_email)]

if "Telefon" in df.columns:
  df["Telefon"] = df["Telefon"].apply(clean_phone)

df = df.drop_duplicates(keep="first")

for key in ("ID", "Email", "Telefon"):
  if key in df.columns:
    df = df.drop_duplicates(subset=[key], keep="first")

for col in ("Datum_upisa", "Datum_diplomiranja"):
  if col in df.columns:
    df[col] = pd.to_datetime(df[col].apply(coerce_date), errors="coerce").dt.date

if {"Datum_upisa", "Datum_diplomiranja"}.issubset(df.columns):
  start = pd.to_datetime(df["Datum_upisa"], errors="coerce")
  end = pd.to_datetime(df["Datum_diplomiranja"], errors="coerce")
  duration_days = (end - start).dt.days
  df["Trajanje_studija"] = (duration_days / 365.25).round(2)

longest = pd.Series(dtype="object")
shortest = pd.Series(dtype="object")
if "Trajanje_studija" in df.columns:
  valid = df[df["Trajanje_studija"].notna()]
  if not valid.empty:
    max_idx = valid["Trajanje_studija"].idxmax()
    min_idx = valid["Trajanje_studija"].idxmin()
    longest = valid.loc[max_idx]
    shortest = valid.loc[min_idx]

print("\nCleaned dataframe head:")
print(df.head(10))

print("\nDuplicates removed, rows/cols:", df.shape)

if not longest.empty:
  print("\nLongest study duration (years):", float(longest["Trajanje_studija"]))
  print(longest.to_dict())

if not shortest.empty:
  print("\nShortest study duration (years):", float(shortest["Trajanje_studija"]))
  print(shortest.to_dict())

out_path = os.path.join(BASE_DIR, "studenti_cleaned.csv")
df.to_csv(out_path, index=False)
print("\nSaved:", out_path)


""" Data visualisation """
import matplotlib.pyplot as plt
import pandas as pd

def col_ok(name: str) -> bool:
  return name in df.columns

if col_ok("Godine"):
  ages = pd.to_numeric(df["Godine"], errors="coerce").dropna()
  plt.figure()
  plt.hist(ages, bins=10)
  plt.title("Age Distribution")
  plt.xlabel("Age")
  plt.ylabel("Count")
  plt.tight_layout()
  plt.show()

if col_ok("Prosek"):
  gpas = pd.to_numeric(df["Prosek"], errors="coerce").dropna()
  if not gpas.empty:
    plt.figure()
    plt.boxplot(gpas, vert=True, showmeans=True)
    plt.title("GPA (Prosek) Distribution")
    plt.ylabel("GPA")
    plt.tight_layout()
    plt.show()

if col_ok("Prosek") and col_ok("Grad"):
  tmp = df.copy()
  tmp["Prosek"] = pd.to_numeric(tmp["Prosek"], errors="coerce")
  avg_by_city = tmp.dropna(subset=["Prosek"]).groupby("Grad")["Prosek"].mean().sort_values(ascending=False)
  if not avg_by_city.empty:
    plt.figure()
    plt.bar(avg_by_city.index.astype(str), avg_by_city.values)
    plt.title("Average GPA by City")
    plt.xlabel("City")
    plt.ylabel("Average GPA")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

if col_ok("ESPB") and col_ok("Grad"):
  tmp = df.copy()
  tmp["ESPB"] = pd.to_numeric(tmp["ESPB"], errors="coerce")
  tmp = tmp.dropna(subset=["ESPB", "Grad"])
  if not tmp.empty:
    order = tmp.groupby("Grad")["ESPB"].median().sort_values(ascending=False).index
    data_by_city = [tmp.loc[tmp["Grad"] == city, "ESPB"].values for city in order]
    plt.figure()
    plt.boxplot(data_by_city, labels=order, showmeans=True)
    plt.title("ESPB Distribution by City")
    plt.xlabel("City")
    plt.ylabel("ESPB")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

if col_ok("Godine") and col_ok("Prosek"):
  x = pd.to_numeric(df["Godine"], errors="coerce")
  y = pd.to_numeric(df["Prosek"], errors="coerce")
  mask = x.notna() & y.notna()
  if mask.any():
    plt.figure()
    plt.scatter(x[mask], y[mask], alpha=0.7)
    plt.title("Age vs GPA")
    plt.xlabel("Age")
    plt.ylabel("GPA")
    plt.tight_layout()
    plt.show()

if col_ok("Prosek") and col_ok("Datum_upisa"):
  tmp = df.copy()
  tmp["Prosek"] = pd.to_numeric(tmp["Prosek"], errors="coerce")
  tmp["Datum_upisa"] = pd.to_datetime(tmp["Datum_upisa"], errors="coerce")
  tmp = tmp.dropna(subset=["Prosek", "Datum_upisa"])
  if not tmp.empty:
    tmp["EnrollmentYear"] = tmp["Datum_upisa"].dt.year
    avg_by_year = tmp.groupby("EnrollmentYear")["Prosek"].mean().sort_index()
    if not avg_by_year.empty:
      plt.figure()
      plt.plot(avg_by_year.index.astype(int), avg_by_year.values, marker="o")
      plt.title("Average GPA by Enrollment Year")
      plt.xlabel("Year")
      plt.ylabel("Average GPA")
      plt.tight_layout()
      plt.show()
