1、Create a folder named CNNNERmodel in the same directory as FGAT12, and place the word embedding set and the BERT pre-trained model into this folder. The word embedding set and the BERT pre-trained model can be downloaded from Tencent AI Lab (https://ai.tencent.com/ailab/nlp/en/download.html) and Hugging Face (https://huggingface.co/google-bert/bert-base-chinese/tree/main)respectively.
    
2、Decide the training dataset by modifying these lines of code in the main1.py file.       
    parser.add_argument('--modelname', default="WeiboNER")    
    parser.add_argument('--savedset', help='Dir of saved data setting', default="data/WeiboNER.dadset")     
    parser.add_argument('--train', default="data/WeiboNER/train.all.bmes")    
    parser.add_argument('--dev', default="data/WeiboNER/dev.all.bmes")     
    parser.add_argument('--test', default="data/WeiboNER/test.all.bmes")    

3、run main1.py

4 paper link:[[paper]](https://link.springer.com/article/10.1007/s10462-024-10891-3?utm_source=rct_congratemailt&utm_medium=email&utm_campaign=oa_20240816&utm_content=10.1007/s10462-024-10891-3)
