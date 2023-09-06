# Instructions for preparing labs 

# Generating stubs for the given implementation

Running example:

```bash
.\venv\Scripts\activate
$env:PYTHONPATH = "$pwd;" + $env:PYTHONPATH
python ./config/generate_stubs/run_generator.py --source_code_path ./lab_4_doc2vec_by_tfidf/main.py --target_code_path ./build/stubs/main_stub.py
```
