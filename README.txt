

I have implemented a BigramModel manually and then implemented the same model using
the NeuralNetwork module. Both of these models have a much higher loss than the Transformer architecture hereby applied. This model uses 6 layer of blocks and each layer is normalized before being fed forward to the next one.
Each block contains a Multi-Head Self Attention mechanism. A single multi-head is formed of 6 heads. 
Each token, representing a character, has an embedding size of 384. We dropout %30 of the neurons during training for a more robust model. All the numerical values may be subject to change.


If you want you can go through the steps and compare both models using the .ipynb file (you can use jupyter notebook for this)
otherwise if you just want to train the Transformer model and generate text you can use 'python TransformerModel.py'

If you don't want to train the model but instead want to generate text using the pre-trained model you need to set "dont_train=True".