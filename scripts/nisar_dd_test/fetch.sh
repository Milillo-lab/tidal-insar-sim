#!/bin/bash
# Download the selected NISAR GUNW products (Earthdata login via ~/.netrc). Resumable (-C -); 3 at a time.
R=${1:-$PWD}            # working directory holding gunw_test_urls.json
cd "$R"
python3 -c "import json;[print(v['url']) for v in json.load(open('gunw_test_urls.json')).values()]" > urls.txt
touch cookies.txt
cd gunw
cat ../urls.txt | xargs -P 3 -I{} nice -n 10 curl -sS -n -L -C - -b ../cookies.txt -c ../cookies.txt --retry 3 -O '{}' 2>>../fetch.err
echo DONE $(date -u +%FT%TZ) >> ../fetch.log
