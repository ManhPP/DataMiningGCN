
# DataMiningGCN  
  

 1. **Run training model GCN & G U-Nets**: trainer.py

	param:   

	 - dataset: Dataset (Cora, CiteSeer, PubMed)  
	 - isUseUNet: Select model Graph U-net or origin GCN  
	 - no: Disables CUDA training  
	 - lr: Initial learning rate  
	 - weight_decay: Weight decay (L2 loss on parameters)  
	 - epochs: Number of epochs to train  
	 - hidden: Number of hidden units  
	 - depth: Depth of Graph U-Nets  
  

 2. **Run training logistic regression + Node2Vec:** node2vec.py
