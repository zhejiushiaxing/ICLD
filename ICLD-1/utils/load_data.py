import random
import pandas as pd



"""简单的数据集类，支持shuffle和select方法"""
class SimpleDataset:
    
    def __init__(self, data):
        self.data = data
    
    """打乱数据集"""
    def shuffle(self, seed=None):
        if seed is not None:
            random.seed(seed)
        shuffled_data = random.sample(self.data, len(self.data))
        return SimpleDataset(shuffled_data)
    
    """选择指定索引的样本"""
    def select(self, indices):
        selected_data = [self.data[i] for i in indices]
        return SimpleDataset(selected_data)
    
    def __len__(self):
        return len(self.data)
    
    def __iter__(self):
        return iter(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]



def load_dataset(dataset_name, dataset_path):
    df = pd.read_parquet(dataset_path)
    dataset = []
    
    if dataset_name == "MMLU":
        for _, row in df.iterrows():
            dataset.append({
                'question': row['question'],
                'choices': row['choices'],
                'answer': row['answer']
            })
    elif dataset_name in ["GSM8K", "NQ"]:
        for _, row in df.iterrows():
            dataset.append({
                'question': row['question'],
                'answer': row['answer']
            })
    elif dataset_name == "BOOLQ":
        for _, row in df.iterrows():
            dataset.append({
                'question': row['question'],
                'answer': row['answer'],
                "passage": row['passage']
            })
    elif dataset_name == "ARC-C":
        for _, row in df.iterrows():
            dataset.append({
                'question': row['question'],
                'choices': row['choices']['text'],
                "answer": row['answerKey']
            })
    elif dataset_name == "PIQA":
        for _, row in df.iterrows():
            dataset.append({
                'question': row['goal'],
                'sol1': row['sol1'],
                'sol2': row['sol2'],
                "answer": row['label']
            })
    elif dataset_name == "MATH500":
        for _, row in df.iterrows():
            dataset.append({
                'question': row['problem'],
                "answer": row['answer']
            })
    elif dataset_name == "HUMANEVAL":
        for _, row in df.iterrows():
            dataset.append({
                'question': row['prompt'],
                "answer": row['canonical_solution']
            })
    elif dataset_name == "MULTILINGUAL":
        for _, row in df.iterrows():
            dataset.append({
                'question': row['inputs'],
                "answer": row['targets'],
                "language": row['language']
            })
    elif dataset_name == "MATHEVAL":
        for _, row in df.iterrows():
            dataset.append({
                'question': row['query'],
                "answer": row['response']
            })
    elif dataset_name == "CODINGEVAL":
        for _, row in df.iterrows():
            dataset.append({
                'question': row['instruction'],
                "answer": row['response']
            })
            
    return SimpleDataset(dataset)