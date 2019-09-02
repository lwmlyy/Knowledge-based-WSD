## Word Sense Disambiguation: A Comprehensive Knowledge Exploitation Framework

This code is an implementation of the Knowledge-based Word Sense Disambiguation (KWSD) Framework that exploits knowledge from different perspectives. In detail, it makes use of knowledge from Wikipedia and WordNet in a similarity-based WSD method and combines the method with a graph-based WSD method in some personalized settings such as top_3 senses filtration, customized sense graph construction and sense importance inheritance. State-of-the-art performance on several standard WSD datasets has proven the effectiveness of such a knowledge-exploitation framework.

### Quick Evaluation
You can make use of the evaluation code provided in [1]. Before that, you should download the evaluation framework on the website [EACL17](http://lcl.uniroma1.it/wsdeval/home). In this resource, you can use the scoror.java to evaluate the performance of our system on different datasets once at a time with the following code, using the files such as 'senseval2.raw.KWSD.key'. 

`java Scorer senseval2/senseval2.gold.key.txt senseval2.raw.KWSD.key`

Also, you can use the evaluation code from UKB website, [UKB](http://ixa2.si.ehu.es/ukb/ukb_3.2.tgz)[2]. Then use **evaluate.sh** to conduct evaluation after you move our result document 'raw.KWSD.key' into ukb-3.2/wsdeval/Keys/raw. The result is supposed to be as follows:

`./evaluate.sh`

Evaluating in Keys  
/* raw.KWSD  
>>  ALL P= 68.0% R= 68.0% F1= 68.0%  
    semeval2007 P= 56.9% R= 56.9% F1= 56.9%  
    semeval2013 P= 68.4% R= 68.4% F1= 68.4%  
    semeval2015 P= 72.3% R= 72.3% F1= 72.3%  
    senseval2 P= 69.6% R= 69.6% F1= 69.6%  
    senseval3 P= 66.1% R= 66.1% F1= 66.1%

#### Reproduction of the system's result
1) Quick: Using the embeddings (say "eLSA01" for senseval2) in the folder, you can run **disambiguation.py** to reproduce the exact reported results. The code itself can evalute the results and also output a file named 'raw.KWSD.key' which can still be evaluated with the above method.
2) Slow: If you want to reproduce the results starting from the domain knowledge document retrieval, it might take a few hours. You also need to download a few documents including British National Corpus([BNC](http://ota.ox.ac.uk/text/2554.zip))[3] and Wikipedia dump for [document retriever](https://github.com/facebookresearch/DrQA) in [4]. The details will be given in the following section.

### Reproduce results from scratch
1. Prepare the 'document retriever' in [4].  Also, you need to download the [TF-IDF model](https://dl.fbaipublicfiles.com/drqa/docs-tfidf-ngram%3D2-hash%3D16777216-tokenizer%3Dsimple.npz.gz) and [Wikipedia](https://dl.fbaipublicfiles.com/drqa/docs.db.gz) database on that website for a quick implementation. We use the model for document name retrieval in **docname_retrieval.py** and use the database to retrieve the corresponding documents in **doc_retrieve.py**. **query_access.py** is to access the query for document retrieval.
2. The retrieved documents are combined with BNC documents which are pre-processed with **bnc_process.py**.The combined document set is then used to learn word representations via lsa in **gensim_lsa.py**.
3. Run **disambiguation.py** for disambiguation of each dataset with the following settings.  
    
    `python disambiguation.py -l True -d domain_doc_name` 

[1] Raganato A.; Camacho-Collados J.; and Navigli R. 2017. Word Sense Disambiguation: A Unified Evaluation Framework and Empirical Comparison. In Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics, 99-110, Valencia, Spain: Association for Computational Linguistics.  
[2] Agirre E.; de Lacalle O. L.; and Soroa A. 2018. The risk of sub-optimal use of Open Source NLP Software: UKB is inadvertently state-of-the-art in knowledge-based WSD. In Proceedings of Workshop for NLP Open Source Software, 29–33, Melbourne, Australia: Association for Computational Linguistics.  
[3] The British National Corpus, version 3 (BNC XML Edition). 2007. Distributed by Bodleian Libraries, University of Oxford, on behalf of the BNC Consortium. URL: http://www.natcorp.ox.ac.uk/
[4] Chen D.; Fisch A.; Weston J.; and Bordes A. 2017. Reading Wikipedia to Answer Open-Domain Questions, In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics, 1870-1879, Vancouver, Canada: Association for Computational Linguistics.# Knowledge-based-WSD
