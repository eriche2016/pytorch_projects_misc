import torchtext.data as data 
import torchtext.datasets as datasets 

text_field = data.Field()
label_field = data.Field(sequential=False,use_vocab=False,postprocessing=data.Pipeline(int))


train = data.TabularDataset(
    path='./test_data_torchtext/qtrain.tsv',format='tsv',fields=[('text', text_field), ('lbl', label_field)], filter_pred=lambda ex: ex.lbl in ['0', '1'])

text_field.build_vocab(train)
# should not call below, because use_vocab is false 
# label_field.build_vocab(train)
