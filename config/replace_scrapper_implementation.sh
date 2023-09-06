set -ex

ls -la lab_5_scrapper/
rm lab_5_scrapper/scrapper.py lab_5_scrapper/scrapper_config.json
ls -la lab_5_scrapper/
mv lab_5_scrapper/scrapper_dynamic.py lab_5_scrapper/scrapper.py
mv lab_5_scrapper/scrapper_dynamic_config.json lab_5_scrapper/scrapper_config.json
ls -la lab_5_scrapper/
