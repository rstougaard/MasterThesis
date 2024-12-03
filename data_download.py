import subprocess
import requests
#import regex as re
#import time

def data_download(source_name):
    # Commands to execute
    source_name_cleaned = source_name.replace(" ", "").replace(".", "dot").replace("+", "plus").replace("-", "minus")
    download_files = ['wget https://fermi.gsfc.nasa.gov/FTP/fermi/data/lat/queries/L2412030422357F1E437D89_SC00.fits',
                        'wget https://fermi.gsfc.nasa.gov/FTP/fermi/data/lat/queries/L2412030422357F1E437D89_PH00.fits',
                        'wget https://fermi.gsfc.nasa.gov/FTP/fermi/data/lat/queries/L2412030422357F1E437D89_PH01.fits',
                        'wget https://fermi.gsfc.nasa.gov/FTP/fermi/data/lat/queries/L2412030422357F1E437D89_PH02.fits',
                        'wget https://fermi.gsfc.nasa.gov/FTP/fermi/data/lat/queries/L2412030422357F1E437D89_PH03.fits',
                        'wget https://fermi.gsfc.nasa.gov/FTP/fermi/data/lat/queries/L2412030422357F1E437D89_PH04.fits',
                        'wget https://fermi.gsfc.nasa.gov/FTP/fermi/data/lat/queries/L2412030422357F1E437D89_PH06.fits',
                        'wget https://fermi.gsfc.nasa.gov/FTP/fermi/data/lat/queries/L2412030422357F1E437D89_PH05.fits'
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

    return

'''
def start_instance(coordfield, timefield, energyfield):
    url = "https://fermi.gsfc.nasa.gov/cgi-bin/ssc/LAT/LATDataQuery.cgi"

    payload = {
        "destination": "query",
        "coordfield": coordfield,
        "coordsystem": "J2000",
        "shapefield": "",
        "timefield": timefield,
        "timetype": "Gregorian",
        "energyfield": energyfield,
        "photonOrExtendedOrNone": "Photon",
        "spacecraft": "on"
    }

    headers = {
        #"Accept-Encoding": "gzip, deflate, br, zstd",
        #"Accept-Language": "da-DK,da;q=0.9,en-US;q=0.8,en;q=0.7",
        #"Content-Type": "multipart/form-data; boundary=----WebKitFormBoundaryjI2qTOFFXNZbyHJ1",
        "Cookie": "_ga=GA1.1.1903984718.1684854647; passphrase=undefined; _ga_CSLL4ZEK4L=GS1.1.1731955458.71.1.1731956207.0.0.0"
        #"Host": "fermi.gsfc.nasa.gov",
        #"Origin": "https://fermi.gsfc.nasa.gov",
        #"Referer": "https://fermi.gsfc.nasa.gov/cgi-bin/ssc/LAT/LATDataQuery.cgi",
        #"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
    }

    response = requests.post(url, data=payload, headers=headers)

    match = re.search(r"The estimated time for your query to complete is (\d+) seconds.*?The results of your query may be found at <a\s*href=\"(.+)\">h", response.text)
    
    estimated_time = match.group(1)
    result_link = match.group(2)

    return estimated_time, result_link
    
def wait_scrape(result_link):
    resp = requests.get(result_link)
    data = resp.text

    pre_content = re.search(r'<pre>(.*?)</pre>', data, re.DOTALL)  # re.DOTALL to match across lines
    if pre_content:
        wget_commands = re.findall(r'wget\s+(https?://\S+)', pre_content.group(1))
        
        # Print all the matched URLs
        for url in wget_commands:
            print(url)

        if len(wget_commands) == 2: # if there is 2 wgets with links. Then we return
            return wget_commands
        
    return None # no result yet. we wait.

def save_to_dir(mappe_navn, url):
    subprocess.run(["mkdir", "-p", mappe_navn])

    name = url.split('/')[-1]
    subprocess.run(['wget', url, '-O', f"./{mappe_navn}/{name}"])



estimated_time, result_link = start_instance(coordfield="NGC1275", timefield="START, END", energyfield="50, 1000000")
print(f"Estimated time: {estimated_time}. Data will be on {result_link}")
while True:
    val = wait_scrape(result_link)

    if val != None:
        a,b = val
        save_to_dir("testtest", a)
        save_to_dir("testtest", b)
        break
    
    print("Not finished... Trying again in 10 sec...")
    time.sleep(10)

#instances = []


#response = requests.get("https://fermi.gsfc.nasa.gov/cgi-bin/ssc/LAT/QueryResults.cgi?id=L2411181504264B1E0A7D72")

#print(response.text)
'''