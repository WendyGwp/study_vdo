def convert_timestamps(input_filename, output_filename, start_time=0):
    timestamps = []
    with open(input_filename, 'r') as input_file:
        for line in input_file.readlines()[20:76]:
            timestamp = int(line.strip())
            timestamps.append(timestamp)

    with open(output_filename, 'w') as output_file:
        for i, timestamp in enumerate(timestamps):
            time_in_seconds = (timestamp - timestamps[0]) * 1e-9 + start_time
            output_file.write(f'{time_in_seconds:.6e}\n')

def main():
    input_filename = 'timestamps.txt'
    output_filename = 'times.txt'
    start_time = 0.0
    convert_timestamps(input_filename, output_filename, start_time)

if __name__ == "__main__":
    main()

