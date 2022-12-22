## Swisstopo
Swisstopo data download is straightforward. For example the following query will return all images captured at 10cm ground sampling distance (other available option is 2m) from Switzerland from January 1st, 2022 to December 5th, 2022.
Note that date_range argument must form a valid json.
```bash
python download_data.py --swisstopo --bbox "[5,46,10,48]" --date_range "[\"2018-01-01\", \"2018-12-31\"]" \
    --resolution 0.1  --save_dir "../out"
```

## Functional Map of the World
Browse and download through the free aws dataset, you need to download everything, it doesn't have a single metdata file that gives information about the locations etc. Requires further processing of the metadata.
- [ ] Preprocess all jsons and build an index.

```bash
aws s3 ls s3://spacenet-dataset/Hosted-Datasets/fmow/fmow-rgb/ --no-sign-request
```

```bash
aws s3 cp s3://spacenet-dataset/Hosted-Datasets/fmow/fmow-rgb/manifest.json.bz2 ./ --no-sign-request
```

## Copernicus API
10m, 20m, 30m bands from the Sentinel-2 can be downloaded via the following command.
```bash
python download_data.py --sentinel --bbox "[5,46,10,48]" --date_range "[\"2022-01-01\", \"2022-01-05\"]" --save_dir "../out"
```
Sentinel data takes more space. 
Another option is to use the [Sentinelsat](https://github.com/sentinelsat/sentinelsat) library, see the [notebook](Sentinelsat.ipynb).


See [here](https://scihub.copernicus.eu/userguide/BatchScripting) on the `dhusget.sh` script and [here](https://scihub.copernicus.eu/twiki/do/view/SciHubUserGuide/OpenSearchAPI) for API options.


## Requirements
### Copernicus
Sentinel API requires authentication, follow the instructions [here](https://scihub.copernicus.eu/userguide/SelfRegistration) to sign up and add the following lines to your `.bashrc`.
```bash
export DHUS_USER="YOUR_USERNAME"
export DHUS_PASSWORD="YOUR_PASSWORD"
export DHUS_URL="https://apihub.copernicus.eu/apihub"
```

### Install GDAL
```bash
conda install -c conda-forge gdal
```

### J2 to Tiff
Possible options exist:
- [Sentinel=Scripts](https://github.com/dairejpwalsh/Sentinel-Scripts.git) tried it with one image, results were not so good.
- [Jp2toTiff](https://github.com/SaifAati/Jp2toTiff/tree/60acdfced438ea8f62f4aa2ea0fc3f36c98cf6cd): haven't tried it
