import urllib.request, os
DATA = r"C:\Users\ohiod\Projects\kappa-sig\domains\networks\data"
os.makedirs(DATA, exist_ok=True)

urls = {
    "bio_yeast.txt": "http://www-personal.umich.edu/~mejn/netdata/power.zip",
    # Try Pajek datasets
    "infra_usair.txt": "http://vlado.fmf.uni-lj.si/pub/networks/data/mix/USAir97.net",
}

# Try multiple BIO sources
bio_urls = [
    ("https://networks.skewed.de/net/moreno_propro/files/moreno_propro.edges", "bio_ppi.txt"),
    ("http://networkrepository.com/bio-yeast-protein-inter.mtx", "bio_yeast.mtx"),
]

for url, fname in bio_urls:
    dest = os.path.join(DATA, fname)
    try:
        print(f"Trying {url[:60]}...")
        urllib.request.urlretrieve(url, dest)
        sz = os.path.getsize(dest)
        print(f"  OK: {sz} bytes")
        if sz > 100: break
    except Exception as e:
        print(f"  FAIL: {e}")

# Try INFRA sources
infra_urls = [
    ("https://networks.skewed.de/net/opsahl-powergrid/files/opsahl-powergrid.edges", "infra_power.txt"),
    ("http://networkrepository.com/power-US-Grid.mtx", "infra_usgrid.mtx"),
]

for url, fname in infra_urls:
    dest = os.path.join(DATA, fname)
    try:
        print(f"Trying {url[:60]}...")
        urllib.request.urlretrieve(url, dest)
        sz = os.path.getsize(dest)
        print(f"  OK: {sz} bytes")
        if sz > 100: break
    except Exception as e:
        print(f"  FAIL: {e}")

# ECO
eco_urls = [
    ("https://networks.skewed.de/net/maayan-foodweb/files/maayan-foodweb.edges", "eco_foodweb.txt"),
]
for url, fname in eco_urls:
    dest = os.path.join(DATA, fname)
    try:
        print(f"Trying {url[:60]}...")
        urllib.request.urlretrieve(url, dest)
        sz = os.path.getsize(dest)
        print(f"  OK: {sz} bytes")
    except Exception as e:
        print(f"  FAIL: {e}")

print("\nDone. Files in", DATA)
for f in os.listdir(DATA):
    fp = os.path.join(DATA, f)
    print(f"  {f}: {os.path.getsize(fp)} bytes")
