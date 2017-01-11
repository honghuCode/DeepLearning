import cPickle
import numpy as np
import os

class Cifar10DataReader():
	def __init__(self,cifar_folder,onehot=True):
		self.cifar_folder=cifar_folder
		self.onehot=onehot
		self.data_index=1
		self.read_next=True
		self.data_label_train=None
		self.data_label_test=None
		self.batch_index=0
		self.W = []
	def unpickle(self,f):
		fo = open(f, 'rb')
		d = cPickle.load(fo)
		fo.close()
		return d
	
	def next_train_data(self,batch_size=10):
		assert 10000%batch_size==0,"10000%batch_size!=0"
		rdata=None
		rlabel=None
		if self.read_next:
			f=os.path.join(self.cifar_folder,"data_batch_%s"%(self.data_index))
			print 'read: %s'%f
			dic_train=self.unpickle(f)
			self.data_label_train=zip(dic_train['data'],dic_train['labels'])#label 0~9 to tuple
			np.random.shuffle(dic_train['labels'])
			
			#print(len(dic_train['labels']))
			
			self.read_next=False
			if self.data_index==5:
				self.data_index=1
			else: 
				self.data_index+=1
		
		if self.batch_index<len(self.data_label_train)//batch_size:
			#print self.batch_index
			datum=self.data_label_train[self.batch_index*batch_size:(self.batch_index+1)*batch_size]
			self.batch_index+=1
			rdata,rlabel=self._decode(datum,self.onehot)
		else:
			self.batch_index=0
			self.read_next=True
			return self.next_train_data(batch_size=batch_size)
			
		return rdata,rlabel
	
	def _decode(self,datum,onehot):
		rdata=[]
		rlabel=[]
		if onehot:
			for d,l in datum:
				rdata.append(np.reshape(np.reshape(d,[3,1024]).T,[32,32,3]))
				#hot=np.zeros(10)
				#hot[int(l)]=1
				#rlabel.append(hot)
				rlabel.append(l)
		else:
			for d,l in datum:
				rdata.append(np.reshape(np.reshape(d,[3,1024]).T,[32,32,3]))
				rlabel.append(int(l))
		return rdata,rlabel
			
	def next_test_data(self,batch_size=100):
		if self.data_label_test is None:
			f=os.path.join(self.cifar_folder,"test_batch")
			print 'read: %s'%f
			dic_test=self.unpickle(f)
			data=dic_test['data']
			labels=dic_test['labels']#0~9
			self.data_label_test=zip(data,labels)
		
		np.random.shuffle(self.data_label_test)
		datum=self.data_label_test[0:batch_size]
		
		return self._decode(datum,self.onehot)
	
	def L(self,x,y,W):
		delta = 10.0
		
		ones = np.ones(x.shape[1])  # generate one row 1
		x = np.row_stack((x,ones))
		scores = W.dot(x)
		#print(W.shape)
		#print(x.shape)
		# compute the margins for all classes in one vector operation
		
		#print(scores.shape)
		
		#margins = np.maximum(0, scores - scores[y] + delta)
		#margins[y] = 0
		#loss_i = np.sum(margins)
		#return loss_i
		#print(type(L))
		loss_i = 0
		scores = scores.T
		i = 0
		for score in scores:
			
			margin = np.maximum(0, score - score[y[i]] + delta)
			margin[y[i]] = 0
			loss_i += np.sum(margin)
		return loss_i

	def classifyBasedStochastic(self,X_train,Y_train):
		bestloss = float("inf") # Python assigns the highest possible float value
		for num in xrange(100000):
			self.W = np.random.randn(10, 3073) * 0.0001 # generate random parameters matrix
			loss = self.L(X_train, Y_train, self.W) # get the loss over the entire training set
			if loss < bestloss: # keep track of the best solution
				bestloss = loss
				bestW = self.W
				print 'in attempt %d the loss was %f, best %f' % (num, loss, bestloss)
		self.W = bestW
	
	def SGDLocal():
		bestloss = float("inf") # Python assigns the highest possible float value
		self.W = np.random.randn(10, 3073) * 0.0001 # generate random parameters matrix
		for num in xrange(100000):
			step_size = 0.0001
			Wtry = self.W + np.random.randn(10, 3073) * step_size
			loss = self.L(X_train, Y_train, Wtry) # get the loss over the entire training set
			if loss < bestloss: # keep track of the best solution
				bestloss = loss
				bestW = self.W
				print 'in attempt %d the loss was %f, best %f' % (num, loss, bestloss)
		self.W = bestW


	
	def predict(self,X_test,W):
		ones = np.ones(X_test.shape[1])  # generate one row 1
		X_test = np.row_stack((X_test,ones))
		scores = self.W.dot(X_test)
		#print(scores)
		scores = scores.T
		preds = []
		for col in scores:
			pred = np.where(col == np.max(col))
			#print(pred[0][0])
			preds.append(pred[0][0])
		return preds
		
		
if __name__=="__main__":
	dr=Cifar10DataReader(cifar_folder="C://cifar-10-python//cifar-10-batches-py")
	import matplotlib.pyplot as plt
	d,l=dr.next_train_data(batch_size=100)
	d = np.asarray(d)
	#print(d.shape)

	d = d.reshape(100,3072)
	#print(d.shape)
	d = d.reshape(3072,100)
	#print(d.shape)
	dr.classifyBasedStochastic(d,l)
	
	
	#get Test Data
	
	
	d,l=dr.next_train_data(batch_size=1000)
	d = np.asarray(d)
	#print(d.shape)

	d = d.reshape(1000,3072)
	#print(d.shape)
	d = d.reshape(3072,1000)
	#print(l)
	preds = dr.predict(d,dr.W)
	
	#print(preds)
	preds = np.array(preds)
	l = np.array(l)
	#print(preds)
	#print("\n")
	
	#print(l)
	#print("\n")
	
	sum = (preds == l)
	
	#print(sum)
	#print("\n")
	rs = []
	for i in sum:
		if i == True:
			k = 1
		else:
			k = 0
		rs.append(k)
	s = np.sum(rs)
	print(s)
	#print(rs)
	
	
	
	'''
	print np.shape(d),np.shape(l)
	plt.imshow(d[0])
	plt.imsave("D://1.jpg",d[0])
	for i in xrange(600):
	  d,l=dr.next_train_data(batch_size=100)
	  print np.shape(d),np.shape(l)
	'''  
	  
	
