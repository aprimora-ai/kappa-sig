import urllib.request, json, csv, os
DATA = r"C:\Users\ohiod\Projects\kappa-sig\domains\networks\data"

# Netzschleuder API format
base = "https://networks.skewed.de/net"

datasets = {
    "bio": f"{base}/moreno_propro",
    "eco": f"{base}/foodweb_baydry",
    "infra": f"{base}/opsahl-powergrid",
}

for name, url in datasets.items():
    try:
        # Get metadata first
        meta_url = url + "/files/metadata.json"
        print(f"[{name}] Trying metadata: {meta_url}")
        req = urllib.request.Request(meta_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            meta = json.loads(resp.read())
            print(f"  Found: {meta}")
    except Exception as e:
        print(f"  Metadata fail: {e}")
        # Try edge list directly
        for ext in [".csv.gz", ".csv", ".edges"]:
            try:
                edge_url = url + "/files/edge_list" + ext
                print(f"  Trying: {edge_url}")
                req = urllib.request.Request(edge_url, headers={"User-Agent": "Mozilla/5.0"})
                dest = os.path.join(DATA, f"{name}{ext}")
                urllib.request.urlretrieve(edge_url, dest)
                sz = os.path.getsize(dest)
                if sz > 100:
                    print(f"  OK: {sz} bytes -> {dest}")
                    break
            except Exception as e2:
                print(f"  {ext} fail: {e2}")
