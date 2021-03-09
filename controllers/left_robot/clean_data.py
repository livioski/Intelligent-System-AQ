with open('data_regression.txt') as f:
    lines = f.read().splitlines()


structured_input = []
structured_output_ang = []
structured_output_dis = []

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

            coord_ang = coord[0]
            coord_dis = coord[1]

            structured_input.append(pixel)
            structured_output_ang.append(coord_ang)
            structured_output_dis.append(coord_dis)
            
            comma += 1


it = 0
new_dic = {}

for inp in structured_input:
    new_dic[f"{inp}"] = [structured_output_ang[it],structured_output_dis[it]]
    it += 1

structured_input = []
structured_output_ang = []
structured_output_dis = []

for key in new_dic:
    new_key = key[1:-1]
    li = list(new_key.split(","))
    new_key = [float(item) for item in li]

    structured_input.append(new_key)
    structured_output_ang.append(new_dic[key][0])
    structured_output_dis.append(new_dic[key][1])

print(structured_input)
