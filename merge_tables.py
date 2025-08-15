import pandas as pd

input_path_1 = "/Users/bowenxia/musicsheets_described.csv"
input_path_2 = "/Users/bowenxia/musicsheets_with_youtube.csv"

df_1 = pd.read_csv(input_path_1)
df_2 = pd.read_csv(input_path_2)

first_four = df_1.iloc[:, :4]
youtube_links = df_2.iloc[:, -1]

df_combined = pd.concat([first_four, youtube_links], axis=1)

df_combined.to_csv("/Users/bowenxia/musicsheets_final.csv", index=False)


