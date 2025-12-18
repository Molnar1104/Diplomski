from datasets import load_dataset

# This loads the dataset in "streaming" mode. 
# It doesn't download the 21GB file to your disk.
dataset = load_dataset("Zihan1004/FNSPID", data_files="Stock_news/nasdaq_exteral_data.csv", split="train", streaming=True)

print("Columns in the dataset:", dataset.features)
print("-" * 20)

# Print the first 5 examples to see what they look like
for i, row in enumerate(dataset):
    if i >= 5: break
    print(row)