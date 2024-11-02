import pandas as pd
import keras
import string
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

def read_files(file, method, class_):
    with open(file, 'r') as f:
        lines = f.readlines()
        for l in lines:
            if l != "\n":
                data =  list(eval(l))
                data.sort(key=lambda tup: tup[1], reverse=True)
                pred[method].append([i[0].replace(' ', '') for i in data if i[1] > 0])

pred = {}

for method in ['lime', 'shap']:
    pred[method] = []
    read_files('explanations_SHAP&LIME/results-class-1-'+method+'.txt', method, 1)


def get_x_rationales(method):
    rationales = []
    for e, p in enumerate(pred[method]):
        rationales.append(' '.join(p))
    
    return rationales

def remove_rationales(method):

    n = math.ceil((np.mean([len(a) for a in pred[method]])))
    new_X = []

    for a, b in zip(pred[method], X):
        
        x = b.split(' ')

        save = []
        i = 0
        while i < n and i < len(a):
            try:
                x.remove(a[i])
                i+=1
            except:
                save.append(a[i])
                i+=1

        b = ' '.join(x)

        for a in save:
            b = b.replace(a, '')
        
        new_X.append(b)

    return new_X 


def predict_fn(texts):
    return model.predict(texts)

X = instances
pred_X = predict_fn(X)

for method in ['lime', 'shap']:
    X_ = remove_rationales(method)
    pred_X_ = predict_fn(X_)

    x_rationales = get_x_rationales(method)
    pred_x_rationales = predict_fn(x_rationales)


    comprehensiveness = []
    for p1, p2 in zip(pred_X, pred_X_):
        comprehensiveness.append(p1[1] - p2[1])

   
    print("Comprehensiveness method {} value {}".format(method, np.mean(comprehensiveness)))

    sufficiency = []

    for p1, p2 in zip(pred_X, pred_x_rationales):
        sufficiency.append(p1[1] - p2[1])

    print("Sufficiency method {} value {}".format(method, np.mean(sufficiency)))
