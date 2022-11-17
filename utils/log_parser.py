import argparse
import pathlib
import sys
if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Parse LEDNet log to get nessesary info. Generate plots out of it.")
    argparser.add_argument("input_path", help="Input log")
    args = argparser.parse_args()

    log_path_obj = pathlib.Path(args.input_path)

    if not log_path_obj.is_file():
        sys.exit(1)

    loaded_log = log_path_obj.read_text()
    log_lines = loaded_log.split("\n")
    filtered_lines = []
    for log_line in log_lines:
        if "INFO" in log_line:
            filtered_lines.append
    sys.exit(0)