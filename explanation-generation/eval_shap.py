import pandas as pd
import keras
import ktrain
from ktrain import text
import shap

dataset = pd.read_csv('/content/sample_data/factnews_dataset_factuality.csv', encoding='UTF-8')
dataset = dataset.sample(frac=1)

#undersampling - deleting samples from the majority class
classe_0 = dataset[dataset.classe == 0]
classe_1 = dataset[dataset.classe == 1]

#Obtaining the less representative sample
sample_0 = classe_0.sample(n=1949, replace=True)
#sample_1 = classe_2.sample(n=336, replace=True)

#Concatenating new data with LESS representativeness into the initial dataset.
dataset_undersampling = pd.concat([classe_1, sample_0])

#bert
(x_train, y_train), (x_test, y_test), preproc = text.texts_from_df(dataset,
                                                                   'sentences',
                                                                   label_columns='classe',
                                                                   maxlen=64,
                                                                   max_features=500,
                                                                   preprocess_mode='bert',
                                                                   lang='pt',
                                                                   val_pct = 0.1,
                                                                   )

#bert classifier
model = text.text_classifier('bert', (x_train, y_train) , preproc=preproc)
classifier = ktrain.get_learner(model,
                             train_data=(x_train, y_train),
                             val_data=(x_test, y_test),
                             batch_size=64
                             )

#load model
predictor = ktrain.load_predictor('factual')
model = ktrain.get_predictor(predictor.model, predictor.preproc)

# instances to explain
instances = pd.read_csv('instances_with_human_rationales/class_1_instances.csv')['Text']

def predict_fn(texts):
    return model.predict(texts)

class_names = ["0", "1"]
masker = shap.maskers.Text(tokenizer=r"\W+")
explainer = shap.Explainer(predict_fn, masker=masker, output_names=class_names)

shap_values = explainer(instances)

with open('/explanations_SHAP&LIME/results-class-1-shap.txt', 'w') as f:
    for i in range(len(instances)):
        f.write('[')
        for j, (word, score) in enumerate(zip(shap_values[i,:, class_names[1]].data, shap_values[i,:, class_names[1]].values)):
            f.write(str((word, score)))    
            if j < len(shap_values[i,:, class_names[1]].data)-1:
                f.write(', ')
        f.write(']\n')