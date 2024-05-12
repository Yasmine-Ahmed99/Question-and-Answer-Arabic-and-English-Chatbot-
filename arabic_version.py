import torch
from sentence_transformers import SentenceTransformer, util
from transformers import BertForQuestionAnswering 
from transformers import AutoTokenizer , AutoModelForQuestionAnswering , AutoModel , TFAutoModel
from transformers import pipeline 
from transformers import ElectraForQuestionAnswering, ElectraForSequenceClassification, AutoTokenizer
from arabert.preprocess import ArabertPreprocessor
import tensorflow as tf

# Load the Arabic text file
def loudtxt(file_path):
 
    with open(file_path, 'r', encoding='utf-8') as file:
        arabic_text = file.read()
    return arabic_text


def splitDoc(arabic_text):
    # split docs into chunks
    chunk_size = 100
    chunks=[]
    for i in range(0, len(arabic_text), chunk_size): chunks .append(arabic_text[i:i+chunk_size] )
    return chunks
    
#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def makeEmbeddings (sentences, q):
    sentences.append(q)
    # Load model from HuggingFace Hub
    embeddings_tokenizer = AutoTokenizer.from_pretrained('medmediani/Arabic-KW-Mdel')
    embeddings_model = AutoModel.from_pretrained('medmediani/Arabic-KW-Mdel')

    # Tokenize sentences
    encoded_input = embeddings_tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = embeddings_model(**encoded_input)

    # Perform pooling. In this case, mean pooling.
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    return sentence_embeddings


def similarity_sentences(sentence_embeddings , sentences ):
    # Calculate cosine similarity between the sentences
    max_cosine_scores=0.0
    for i in range(len(sentence_embeddings)-1):
        cosine_scores = util.pytorch_cos_sim(sentence_embeddings[i], sentence_embeddings[len(sentence_embeddings)-1])
        if (cosine_scores.item()> max_cosine_scores): 
            max_cosine_scores=cosine_scores.item() 
            index_sentence =i

    
    return  max_cosine_scores , sentences[index_sentence]



def askQs (q , sentence ):
    # Load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained("wonfs/arabert-v2-qa")
    model = AutoModelForQuestionAnswering.from_pretrained("wonfs/arabert-v2-qa", from_tf=True)

    

    # Tokenize the question and the context
    inputs = tokenizer.encode_plus(q, sentence, return_tensors="pt", max_length=512, truncation=True)

    # Perform question-answering
    outputs = model(**inputs)
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    # Get the most likely answer
    start_index = start_scores.argmax(dim=1).item()
    end_index = end_scores.argmax(dim=1).item()

    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_index:end_index+1]))
    return answer

q= " ما هي أدوات السيو؟"
text = loudtxt('lougehrig.txt')
#chunks=splitDoc(text)
l_text=[]
l_text.append(text)
sentence_embeddings=makeEmbeddings (l_text, q)
max_cosine_scores , sentence=similarity_sentences(sentence_embeddings , l_text )
answer=askQs (q , sentence )
print(answer)