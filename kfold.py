splits = [3, 5, 7, 10]
for k in splits:
	# create the model 
	# compile the model
	split_val = len(train_feat)/10
	valrows=([x for x in range(split_val*(k-3),split_val*(k))])
	trainrows=([x for x in train_feat if x not in valrows])
	model.fit(train_feat[trainrows,:], train_targets[trainrows,:]…, 				
	validation_data=(train_feat[valrows,:], y_train[valrows,:] …)
