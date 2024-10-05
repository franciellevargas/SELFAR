<h2 align="center"> SELFAR - Sentence-Level Factual Reasoing To Improve Explainable Fact-Checking in Portuguese </h2>  

</br>
<p align="justify"> Most existing fact-checking systems are unable to explain their decisions by providing relevant rationales (justifications) for their predictions. It highlights a lack of transparency that poses significant risks, such as the prevalence of unexpected biases, which may increase political polarization due to limitations in impartiality. To address this critical gap, we introduce a new method to improve explainable fact-checking. The SEntence-Level FActual Reasoning (SELFAR) relies on fact extraction and verification by predicting the news source reliability and factuality (veracity) of news articles or claims at the sentence level, generating post-hoc explanations using SHAP/LIME and zero-shot prompts. Our experiments show that unreliable news stories predominantly consist of subjective statements, in contrast to reliable ones. Consequently, predicting unreliable news articles at the sentence level by analyzing impartiality and subjectivity is a promising approach for fact extraction and improving explainable fact-checking. Furthermore, LIME outperforms SHAP in explaining predictions on reliability. Additionally, while zero-shot prompts provide highly readable explanations and achieve an accuracy of 0.71 in predicting factuality, their tendency to hallucinate remains a challenge. Lastly, this paper also introduces the first study on explainable fact-checking for the Portuguese language. The SELFAR to enhance explainable fact-checking. SELFAR encompasses three main tasks: <b>Fact Extraction (FE)</b>, <b>Fact Verification (FV)</b>, and <b>Explanation Generation (EG)</b>, as shown in figure as follows.
</p> 


 ![SSC-logo-300x171](https://github.com/franciellevargas/franciellevargas.github.io/blob/5a2d7baf37291cc83a10632b11c3341e44358fe7/img/selfar.png)

 
<h2 align="left"> CITING </h2>

<p align="justify">
Francielle Vargas, Isadora Salles, Diego Alves, Ameeta Agrawal, Thiago Pardo, Fabrício Benevenuto (2024). <b>Improving Explainable Fact-Checking via Sentence-Level Factual Reasoning</b>. Proceedings of the 7th Fact Extraction and VERification Workshop (FEVER @ EMNLP 2024), Miami, United States. 
</p>
