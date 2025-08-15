import pandas as pd

INPUT_CSV  = "/Users/bowenxia/musicsheets.csv"
OUTPUT_CSV = "/Users/bowenxia/musicsheets_described.csv"
TITLE_COL  = "title"
ARTIST_COL = "artists"

df = pd.read_csv(INPUT_CSV)

