import torch

from openprompt.data_utils import InputExample

#相关和不相关
classes = [ # 相关代表相关
    "Relevant",
    "Irrelevant"
]
dataset = [
    # text_a is the input text of the data, some other datasets may have multiple input sentences in one example.
    InputExample(
        guid = 0,
        text_a = "I hate watching horror movies.",
    ),
    InputExample(
        guid = 1,
        text_a = "The film was badly made.",
    ),
    InputExample(
        guid=2,
        text_a="I love movie",
    ),
    InputExample(
        guid=3,
        text_a="The ending of ’The Shawshank Redemptionis‘ my favorite.",
    ),
    InputExample(
        guid=4,
        text_a="’The Fast and the Furious‘ franchise has already released ten movies.",
    ),
    InputExample(
        guid=5,
        text_a="I had a stir-fried pork dish last night.",
    ),
    InputExample(
        guid=6,
        text_a="I went to see the dentist yesterday.",
    ),
    InputExample(
        guid=7,
        text_a="The capital of the United States is Washington D.C.",
    ),
    InputExample(
        guid=8,
        text_a="There are so many dialects in China.",
    ),
    InputExample(
        guid=9,
        text_a="I'm worried about tomorrow's English CET-6 exam.",
    ),

]


from openprompt.plms import load_plm
plm, tokenizer, model_config, WrapperClass = load_plm("bert", "bert-base-cased")

from openprompt.prompts import ManualTemplate
promptTemplate = ManualTemplate(
    text = ' {"placeholder":"text_a"} topic is {"mask"}',
    tokenizer = tokenizer,
)


from openprompt.prompts import ManualVerbalizer
promptVerbalizer = ManualVerbalizer(
    classes = classes,
    label_words = {
        "Relevant": ["film","movie"],
        "Irrelevant": ["others"],
    },
    tokenizer = tokenizer,
)

from openprompt import PromptForClassification
promptModel = PromptForClassification(
    template = promptTemplate,
    plm = plm,
    verbalizer = promptVerbalizer,
)


from openprompt import PromptDataLoader
data_loader = PromptDataLoader(
    dataset = dataset,
    tokenizer = tokenizer,
    template = promptTemplate,
    tokenizer_wrapper_class=WrapperClass,
)

# 第七步：训练和预测：完成了!我们可以像Pytorch中的其他过程一样进行训练和推理。
# making zero-shot inference using pretrained MLM(masked language model) with prompt
promptModel.eval()
with torch.no_grad():
    for batch in data_loader:
        logits = promptModel(batch)
        preds = torch.argmax(logits, dim=-1)
        print(classes[preds])
# predictions would be 1, 0 for classes 'positive', 'negative'


