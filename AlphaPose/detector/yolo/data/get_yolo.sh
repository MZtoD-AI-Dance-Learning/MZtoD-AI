# Field = 1VdeLMaqKxT2gITwIqgsF6h3u50HZjsG_
# Name
wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1VdeLMaqKxT2gITwIqgsF6h3u50HZjsG_' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1VdeLMaqKxT2gITwIqgsF6h3u50HZjsG_" -O yolov3-spp.weights && rm -rf ~/cookies.txt