from __future__ import print_function
import os
import json
from flair.data import Sentence
from flair.models import SequenceTagger

# We need to define 4 functions which loads the model, transforms input, predicts and transforms output.

def model_fn(model_dir):
    # Load the model using Flair sequence tagger.
    print(os.path.join(model_dir, 'model.pt'))
    model = SequenceTagger.load(os.path.join(model_dir, 'model.pt'))
    return model

def input_fn(request_body, request_content_type):
    print("printing request body")
    print(request_body)
    if request_content_type == 'application/json':
        jsonobj = json.loads(request_body)
        print(jsonobj["text"])
        return jsonobj["text"]
    else:
        # Handle other content-types here or raise an Exception
        # if the content type is not supported.
        pass
    
def predict_fn(input_data, model):
    # create example sentence
    print("inside predict function")
    print(input_data)
    print(model)
    sentence = Sentence(input_data)
    print(sentence)
    model.predict(sentence)
    print("printing prediction")
    print(sentence)
    #predict tags and print
    return sentence
    

def output_fn(prediction, content_type):
    out_array = []
    if prediction:
        for entity in prediction.get_spans('ner'):
            max_score = 0
            label_value= ''
            out = entity.to_dict()
            for label in out["labels"]:
                if label.score >= max_score:
                    max_score = label.score
                    label_value = label.value
            entDict = {}
            entDict["text"] = out["text"]
            entDict["start_pos"] = out["start_pos"]
            entDict["end_pos"] = out["end_pos"]
            entDict["label"] = label_value
            entDict["confidence"] = max_score
            out_array.append(entDict)
    return json.dumps({"entities":out_array}),content_type