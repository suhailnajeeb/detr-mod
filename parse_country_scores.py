from prettytable import PrettyTable

table_name = 'nms_models.csv'
#output_path = 'slurm-43027888.out'
output_path = 'slurm-43168182.out'

with open(output_path, 'r') as f:
    output = f.readlines()

table = PrettyTable(['Country', 'mAP', 'mAP50'])

for line in output:
    if "Processing data for:" in line: 
        country = line.split(': ')[1][:-1]
    if 'Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]' in line:
        map50 = line[-6:-1]
    if 'Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ]' in line:
        mapAll = line[-6:-1]
        table.add_row([country, mapAll, map50])

print(table)

# Save th table to a CSV file: 
with open(table_name, 'w') as f: 
    f.write(table.get_csv_string())