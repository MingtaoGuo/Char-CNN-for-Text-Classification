# Char-CNN-for-Text-Classification
Character-level Convolutional Networks for Text Classification  

# Introduction
This code mainly implement the paper [Character-level Convolutional Networks for Text Classification ](https://arxiv.org/abs/1509.01626)(Char CNN)
![](https://github.com/MingtaoGuo/Char-CNN-for-Text-Classification/blob/master/IMAGES/model.jpg)

I find this image rather difficult to understand. So I redrawn the following image according to my own understanding.
![](https://github.com/MingtaoGuo/Char-CNN-for-Text-Classification/blob/master/IMAGES/MODEL_GMT.jpg)
In actually, Char CNN is the same as conventional CNN (for image).
# Dataset
Please click [HERE](https://github.com/mhjabreel/CharCNN/tree/master/data/ag_news_csv) to download the 'train.csv' and 'test.csv'.
The original dataset [AG's corpus of news articles](http://www.di.unipi.it/~gulli/AG_corpus_of_news_articles.html)
Data sample:

|class|text content|
|-|-|
|world|Sister of man who died in Vancouver police custody slams chief (Canadian Press). Canadian Press - VANCOUVER (CP) - The sister of a man who died after a violent confrontation with police has demanded the city's chief constable resign for defending the officer involved.|
|sports|Johnson Back to His Best as D-Backs End Streak. NEW YORK (Reuters) - Randy Johnson struck out 14 batters in  8 1/3 innings to help the Arizona Diamondbacks end a nine-game  losing streak with a 2-0 win over the host New York Mets in the  National League Sunday.|
|Business|Dollar Briefly Hits 4-Wk Low Vs Euro.  LONDON (Reuters) - The dollar dipped to a four-week low  against the euro on Monday before rising slightly on  profit-taking, but steep oil prices and weak U.S. data  continued to fan worries about the health of the world's  largest economy.|
|Sci/Tech|Search providers seek video, find challenges. Internet search providers are reacting to users #39; rising interest in finding video content on the Web, while acknowledging that there are steep challenges that need to be overcome.|
# Requirements
1. python3.5
2. tensorflow1.4.0
3. numpy
4. pandas
# Results
|Loss|Training accuracy|
|-|-|
|![](https://github.com/MingtaoGuo/Char-CNN-for-Text-Classification/blob/master/IMAGES/loss.jpg)|![](https://github.com/MingtaoGuo/Char-CNN-for-Text-Classification/blob/master/IMAGES/acc.jpg)|
# Acknowledgement
Thanks for [mhjabreel](https://github.com/mhjabreel)' csv dataset
# Reference
[1] 
