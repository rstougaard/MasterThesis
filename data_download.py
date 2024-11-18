import subprocess

def data_download(source_name):
    # Commands to execute
    source_name_cleaned = source_name.replace(" ", "").replace(".", "dot").replace("+", "plus").replace("-", "minus")
    download_files = [
        'wget https://fermi.gsfc.nasa.gov/FTP/fermi/data/lat/queries/L241116043736D7E8FD4F25_SC00.fits',
        'wget https://fermi.gsfc.nasa.gov/FTP/fermi/data/lat/queries/L241116043736D7E8FD4F25_PH00.fits'
    ]
    move_SCfiles = 'mv *SC00.fits SC.fits'
    move_datafiles = f'mv *.fits ./data/{source_name_cleaned}'
    ls = f'ls ./data/{source_name_cleaned}/*PH*.fits > ./data/{source_name_cleaned}/events.list'
    cat = f'cat ./data/{source_name_cleaned}/events.list'

    # Run commands
    for cmd in download_files:
        subprocess.run(cmd, shell=True, check=True, executable='/bin/bash')

    subprocess.run(move_SCfiles, shell=True, check=True, executable='/bin/bash')
    subprocess.run(move_datafiles, shell=True, check=True, executable='/bin/bash')
    subprocess.run(ls, shell=True, check=True, executable='/bin/bash')
    subprocess.run(cat, shell=True, check=True, executable='/bin/bash')
