1、在FGAT12同级目录下新建CNNNERmodel文件夹，并将词向量集和BERT预训练模型放到该文件夹，词向量集和BERT预训练模型可分别通过https://ai.tencent.com/ailab/nlp/en/download.html和https://huggingface.co/google-bert/bert-base-chinese/tree/main下载。
2、通过修改main1.py文件下的这几行代码，决定训练的数据集。
    parser.add_argument('--modelname', default="WeiboNER")
    parser.add_argument('--savedset', help='Dir of saved data setting', default="data/WeiboNER.dadset")
    parser.add_argument('--train', default="data/WeiboNER/train.all.bmes")
    parser.add_argument('--dev', default="data/WeiboNER/dev.all.bmes")
    parser.add_argument('--test', default="data/WeiboNER/test.all.bmes")
3、运行main1.py文件。
