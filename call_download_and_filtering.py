import argparse
import json
from data_download import data_download
from data_filtering import data_filtering
from binned_analysis import check_paths

def load_settings(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description='Run AGN analysis with custom settings')
    parser.add_argument('--settings', required=True, help='Path to the JSON file containing AGN settings')
    args = parser.parse_args()

    # Load settings from JSON file
    settings = load_settings(args.settings)
    print(settings)

    #check paths for likelihood analysis
    check_paths(settings['source_name'], settings['time_interval_name'])

    # Run the analysis with the loaded settings
    data_download(settings['source_name'])
    
    data_filtering(settings['source_name'], settings['start_time'], settings['end_time'], settings['minimal_energy'], settings['maximal_energy'], settings['ra'], settings['dec'], settings['time_interval_name'])

if __name__ == "__main__":
    main()