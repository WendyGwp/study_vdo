# -*- coding: utf-8 -*-
import os
import argparse
def convert_timestamps(input_filename, output_filename, start_time,start_line ,end_line ):
    timestamps = []
    with open(input_filename, 'r') as input_file:
        for line in input_file.readlines()[start_line:end_line]:
            timestamp = int(line.strip())
            timestamps.append(timestamp)

    with open(output_filename, 'w') as output_file:
        for i, timestamp in enumerate(timestamps):
            time_in_seconds = (timestamp - timestamps[0]) * 1e-4 + start_time
            output_file.write(f'{time_in_seconds:.6e}\n')

def main(folder_path,start_line,end_line):
    input_filename = folder_path + 'timestamps.txt'
    output_filename = folder_path + 'times.txt'
    start_time = 0.0
    convert_timestamps(input_filename, output_filename, start_time,start_line,end_line)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="times")
    parser.add_argument("folder_path", type=str, help="orign times in which fileFloder")
    parser.add_argument("start", type=int, help="orign times in which fileFloder")
    parser.add_argument("end", type=int, help="orign times in which fileFloder")
    args = parser.parse_args()
    main(args.folder_path,args.start,args.end)

