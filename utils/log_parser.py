import argparse
import pathlib
import sys
import datetime
import re
import json
if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Parse LEDNet log to get nessesary info. Generate plots out of it.")
    argparser.add_argument("input_path", help="Input log")
    args = argparser.parse_args()

    log_path_obj = pathlib.Path(args.input_path)

    if not log_path_obj.is_file():
        sys.exit(1)

    value_dict = {}
    loaded_log = log_path_obj.read_text()
    log_lines = loaded_log.split("\n")
    filtered_lines = []
    for log_line in log_lines[1:]:
        if all(x in log_line for x in ["INFO", "epoch"]) and not any(x in log_line for x in ["Resuming", "Start"]):
            filtered_lines.append(log_line)
    filtered_lines = filtered_lines[0:]
    for log_line in filtered_lines:
        # split on integers
        splitted_line = re.split('(\[.*\])', log_line)
        trimmed = [x.strip() for x in splitted_line]
        iter_num = int(re.split('(:)', splitted_line[1])[4].strip().replace(",", "")[:-3])
        lr = float(re.split('(:)', splitted_line[1])[6].split(",)]")[0][1:])
        l_pix = float(re.split('(:)', splitted_line[2])[2].split(" ")[1])
        l_side_pix = float(re.split('(:)', splitted_line[2])[4].split(" ")[1])
        l_percep = float(re.split('(:)', splitted_line[2])[6].split(" ")[1])
        l_side_percep = float(re.split('(:)', splitted_line[2])[8].split(" ")[1])
        value_dict.update({iter_num: {"lr": lr, "l_pix": l_pix, "l_side_pix": l_side_pix, "l_percep": l_percep, "l_side_percep": l_side_percep}})

    with open(f"stats_{datetime.datetime.now()}.json", "w") as f:
        json_out = json.dumps(value_dict)
        f.write(json_out)
    sys.exit(0)