import urllib.request as r
import progressbar
import os
import tarfile
files = dict()

'''
This downloads all the articles from a given link
'''
files['comm_use.0-9A-B.txt.tar.gz'] = "ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/comm_use.0-9A-B.txt.tar.gz"
#files['comm_use.C-H.txt.tar.gz'] = "ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/comm_use.C-H.txt.tar.gz"
#files['comm_use.I-N.txt.tar.gz']="ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/comm_use.I-N.txt.tar.gz"
#files['comm_use.O-Z.txt.tar.gz'] = "ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/comm_use.O-Z.txt.tar.gz"

for key in files:
    u = r.urlopen(files[key], key)
    store = 'data'
    pathJoined = os.path.join(store, key)
    if not os.path.exists(store):
        os.makedirs(store)
    f = open(pathJoined, 'wb')
    meta = u.info()
    file_size = int(meta._headers[1][1])
    print("Downloading: %s Bytes: %s" % (key, file_size))
    file_size_dl = 0
    block_sz = 1572864
    with progressbar.ProgressBar(max_value=file_size) as bar:
        while True:
            buffer = u.read(block_sz)
            if not buffer:
                break
            file_size_dl += len(buffer)
            f.write(buffer)
            bar.update(file_size_dl)
        f.close()
        
for root, dirs, files in os.walk('data'):
    for file in files:
        pathJoined = os.path.join(root, file)
        if (file.endswith("tar.gz")):
            tar = tarfile.open(pathJoined, "r:gz")
            tar.extractall(os.path.join('data', file.split('tar.gz')[0]))
            tar.close()