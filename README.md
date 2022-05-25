# Neural-networks

**ANUHYA KALVAKALA**

Creating and evaluating neural networks using Keras Library

**First Dataset**

**Introduction:**

1.	The dataset is mainly to differentiate between the Nasal and the oral vowels.
2.	This comes from 1809 isolated syllables.
3.	Five are attributes were chosen to represent them V1, V2, V3, V4, V5.
4.	Target is given as two classes (1,2).
5.	Total no of instances is 5404 X 5.

**Nodes	Accuracy	P value**

![image](https://user-images.githubusercontent.com/96926526/170377827-b9095456-9edf-421f-abac-bbc745efd6d3.png)


**Discussion:**

**No hidden layers:**

1.	we had connected our input with the 10 units with the activation function relu and output of 2 nodes with activation as SoftMax
2.	We used epochs of 10,100,200,400 and observed the loss was constant after 100 and accuracy was maintained at 83 hence, we considered as epochs as    100.
3.	Loss was started at 0.6644 in fold 1 epoch1 and ended with 0.3762 at fold10 in epoch 100

**Less dense one hidden layer:**

1.	we had connected our input with the 10 units with the activation function relu, output of 2 nodes with activation as SoftMax and in our hidden layer we gave 15 units.
2.	We used epochs of 10,100,200,400 and observed the loss was constant after 100 and accuracy was maintained at 83 hence, we considered as epochs as 100.
3.	Loss was started at 0.6818 in fold 1 epoch1 and ended with 0.3608 at fold10 in epoch 100
4.	
**High Dense one hidden layer:**

1.	we had connected our input with the 10 units with the activation function relu, output of 2 nodes with activation as SoftMax and in our hidden layer we gave 15 units.
2.	We used epochs of 10,100,200,400 and observed the loss was constant after 100 and accuracy was maintained at 83 hence, we considered as epochs as 100.
3.	Loss was started at 0.6063 in fold 1 epoch1 and ended with 0.3587 at fold10 in epoch 100


**Two hidden layers:**

1.	we had connected our input with the 10 units with the activation function relu and output of 2 nodes with activation as SoftMax
2.	We used epochs of 10,100,200,400 and observed the loss was constant after 100 and accuracy was maintained at 83 hence, we considered as epochs as 100.
3.	Loss was started at 0.5479 in fold 1 epoch1 and ended with 0.3941 at fold10 in epoch 100

Our one hidden layer with dense nodes of 16 worked best out of all with accuracy of **0.849371874332428**



**Second Dataset**

**Introduction:**

1.	The dataset is mainly to differentiate between the shapes of bananas.
2.	three are attributes were chosen to represent them V1, V2.
3.	Target is given as two classes. (1,2)
4.	Total no of instances is 5300 X 2.


**Nodes	Accuracy	P value**

![image](https://user-images.githubusercontent.com/96926526/170377954-03f70b94-ad79-41e7-b49c-190f9862a7f9.png)


**Discussion:**

**No hidden layers:**

1.	we had connected our input with the 10 units with the activation function relu and output of 2 nodes with activation as SoftMax
2.	We used epochs of 10,100,200,400 and observed the loss was constant after 100 and accuracy was maintained at 83 hence, we considered as epochs as 100.
3.	Loss was started at 0.6869 in fold 1 epoch1 and ended with 0.3201 at fold10 in epoch 100

**Less dense one hidden layer:**

1.	we had connected our input with the 10 units with the activation function relu, output of 2 nodes with activation as SoftMax and in our hidden layer we gave 15 units.
2.	We used epochs of 10,100,200,400 and observed the loss was constant after 100 and accuracy was maintained at 83 hence, we considered as epochs as 100.
3.	Loss was started at 0.6485 in fold 1 epoch1 and ended with 0.2975 at fold10 in epoch 100

**High Dense one hidden layer:**

1.	we had connected our input with the 10 units with the activation function relu, output of 2 nodes with activation as SoftMax and in our hidden layer we gave 15 units.
2.	We used epochs of 10,100,200,400 and observed the loss was constant after 100 and accuracy was maintained at 83 hence, we considered as epochs as 100.
3.	Loss was started at 0.6558 in fold 1 epoch1 and ended with 0.2748 at fold10 in epoch 100


**Two hidden layers:**

1.	we had connected our input with the 10 units with the activation function relu and output of 2 nodes with activation as SoftMax
2.	We used epochs of 10,100,200,400 and observed the loss was constant after 100 and accuracy was maintained at 90 hence, we considered as epochs as 100.
3.	Loss was started at 0.6891 in fold 1 epoch1 and ended with 0.2794 at fold10 in epoch 100

Our one hidden layer with dense nodes of 16 worked best out of all with accuracy of **0.9001886785030365**

