import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing

with open('data_regression.txt') as f:
    lines = f.read().splitlines()

structured_input = []
structured_output = []

model = LinearRegression()

for line in lines:

    split_at = 0
    comma = 0

    for char in line:

        split_at += 1

        if char == "," and comma == 0:
            comma += 1

        elif char == "," and comma == 1:

            pixel = line[2:split_at-2]
            pixel = list(pixel.split(",")) 
            pixel = [float(item) for item in pixel]

            coord = line[split_at+2:-2]
            coord = list(coord.split(",")) 
            coord = [float(item) for item in coord]

            structured_input.append(pixel)
            structured_output.append(coord)
            
            comma += 1

structured_input_norma = preprocessing.normalize(structured_input)
model.fit(structured_input_norma, structured_output)

r_sq = model.score(structured_input_norma, structured_output)
            
print(f"lo score è {r_sq}")         