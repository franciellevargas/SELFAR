import pandas as pd
import keras
import ktrain
from ktrain import text
from lime.lime_text import LimeTextExplainer

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
explainer = LimeTextExplainer(class_names = class_names)

explainers = []
for i in instances:
    t0 = time.time()
    exp = explainer.explain_instance(i, predict_fn, num_features = 10, labels=(1,)) 
    explainers.append(exp)


with open('/explanations_SHAP&LIME/results-class-1-lime.txt', 'w') as f:
    for exp in explainers:
        f.write(str(exp.as_list()))
        f.write('\n')
