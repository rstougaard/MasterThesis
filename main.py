import argparse
import json
from binned_analysis import check_paths, run_analysis

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
    run_analysis(settings['source_name'], settings['short_name'], settings['num_workers'], settings['num_time_intervals'], 
                 settings['time_interval_name'], settings['start_month'], settings['ra'], settings['dec'], settings['minimal_energy'], settings['maximal_energy'])
    
if __name__ == "__main__":
    main()