import requests
import pandas as pd

url = "https://api.gdc.cancer.gov/files"

params = {
    "filters": '{"op":"and","content":[{"op":"=","content":{"field":"cases.project.project_id","value":"TCGA-OV"}},{"op":"=","content":{"field":"data_type","value":"Gene Expression Quantification"}},{"op":"=","content":{"field":"analysis.workflow_type","value":"HTSeq - FPKM"}}]}',
    "fields": "file_id,file_name,cases.submitter_id,cases.disease_type",
    "format": "JSON",
    "size": "2000"
}
response = requests.get(url, params=params)
data = response.json()


files = pd.DataFrame([f["file_id"] for f in data["data"]["hits"]], columns = ["file_id"])

print("Total files:" , len(files))
