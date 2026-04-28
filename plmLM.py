#plm and testing
from transformers import BertModel, BertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, matthews_corrcoef
import torch
#misc
from Bio import SeqIO
import pandas as pd
from tqdm import tqdm

#Converts file into dataframe
def fastaConverter(filename):
    dataList = [] 
    with open(filename,"r") as handle:
        for entry in SeqIO.parse(handle, "fasta"):
             dataList.append({
                    'label': int(entry.id[8]),
                    'sequence': str(entry.seq),
                    'length': len(entry.seq)})
    dataframe = pd.DataFrame(dataList)
    return dataframe

###Load data in, training/testing.  Split later for validation.  
trainingData = fastaConverter('Train.fasta')
testingData = fastaConverter('test.fasta')


#load pbert model
tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
model = BertModel.from_pretrained("Rostlab/prot_bert").to('cuda' if torch.cuda.is_available() else 'cpu')
model.eval()
def proteinbert_feature_extractor(peptides, batch_size=32):
      features = []
      for i in tqdm(range(0, len(peptides), batch_size)):
        batch_seqs = peptides[i : i + batch_size]
        
        # Format: "ACDEF" -> "A C D E F", necessary for pbert huggingface
        spaced_seqs = [" ".join(list(seq)) for seq in batch_seqs]
        
        # Tokenize
        inputs = tokenizer(spaced_seqs, return_tensors='pt', padding=True, truncation=True, max_length=52)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Use the index 0 as summary
            batch_features = outputs.last_hidden_state[:, 0, :].cpu()
            features.append(batch_features)
            
      return torch.cat(features, dim=0)

def knn_plm(data):
    peptide_features = proteinbert_feature_extractor(data['sequence'])
    pep_labels = torch.tensor(data['label'].values).float().unsqueeze(1)
    X = peptide_features.numpy() 
    y = data['label'].values

    # 2. Split into training and testing (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. K-fold validation KNN
    grid_search = GridSearchCV(KNeighborsClassifier(weights='distance'), {'n_neighbors': range(1, (int(6387**(1/2))))}, cv=5)
    grid_search.fit(X_train, y_train)

    # 4. Evaluate
    y_pred = grid_search.predict(X_test)
    print(f"Best k param: {grid_search.best_params_} best score: {grid_search.best_score_}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(classification_report(y_test, y_pred))
    print(f"Matthews correlation coefficient(MCC): {matthews_corrcoef(y_test, y_pred)}")
    
knn_plm(trainingData)
