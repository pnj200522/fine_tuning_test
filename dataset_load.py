from datasets import load_dataset

dataset = load_dataset("Abirate/english_quotes")  # 예제 데이터셋

# CSV 또는 JSON 형태의 파일을 사용하는 경우.
# dataset = load_dataset("csv", data_files="your_dataset.csv")

print(dataset["train"][0])  # 데이터 확인
