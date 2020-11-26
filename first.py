import torch #importe la lib Pytorch
import numpy #  
print(torch.__version__) #donne la v de torch
print(torch.cuda.is_available()) #check si cuda est compatible avec mon gpu
print(torch.version.cuda) #v de cuda


#test pour les tensor et cuda

t=torch.tensor([1,1,4]) #tensor de 1D (vector)
print(t)
#pour le trensfere vert le gpu on fait
t=t.cuda()
print(t)
#les fondamtale sont RAMK et AXES et shape
#rank= les dimansion du tensor
#le axis suit le RANK dison que on a 2 RANk donc on a 2 axes (x,y) la pour le premier axes il contien des array pour le secon des number
#le shape est le meme que le size dans pytorch comme pour une tensor 3x3 le shape est 3,3
#reshaping pour modifie la forme dison de 6x1 a 3x2
#exemple
a=[
    [1,2,4],
    [5,4,7],
    [6,4,1],
    [4,4,7]
    ]
k=torch.tensor(a)
print(k.shape)
k=k.reshape(1,12)
print(k.shape)
#on CNN le input est coupe on 4 truck [B,C,W,H] B=batch C=chanel(R,G,B) W=with H=heign
#on CNN le output dune convulution est apeller feature map

print(t.dtype)#type of data in tensor
print(t.device)#ou il run (la device use for this tensor)
print(t.layout)#cote array
#on peut specifie la device qui sera la premier (default)
device=torch.device('cuda:0')
print(device)
#il ya 4 type de creation de tensor dans pytorch
#torch.Tensor(data) --->class constructeur (il cree des double)
#torch.tensor(data) --->factory function (methode)
#torch.as_tensor(data)
#torch.from_numpy(data)

#les quelque fonction pour torch sont
#torch.eye() ---> un tensor identity 
#torch.zeros() --> une tensor de 0 
#torch.ones()  ---> une tensor de 1 
#torch.rand() ----> une tensor de valeur entre [0-1]

#torch.get_default_dtype() donne par defaut le type torch.float32
#torch.tensor(data,dtype=type) pour definir un type

#torch.Tensor(data) et torch.tensor(data) on la data original (il cree une copy)
#torch.as_tensor(data) et torch.from_numpy(data) on la reference a la data avec numpy (si la data est modifi il ce modifie)


#les 4 opperation principale du tensor sont
#1)Reshaping operations ->change le shape du tensor (size)
#2)element-wise operations
#3)reduction operation
#4)access operation

#on peut optenir le shape soit avec atribut shape ou methode size

#pour le produit on utilise la methode prod ex: t.prod() parfois sa renfois au scalaire du componette
#ya aussi la methode numel qui sinifie number of element

#on peux  reshape avec methode squeeze et unsqueeze
#squeeze retire tous les axes qui on un size de 1
#unsqueeze ajoute un axes avec un size de 1
print(k.shape)
print(k.squeeze())
print(k.squeeze().shape)
print(k.squeeze().unsqueeze(dim=0))
print(k.squeeze().unsqueeze(dim=0).shape)
#flatten a tensor ceut dire que on cree un array de 1D (1D tensor)
#on peux flatter une fonction on utilise largument -1 pytorch comprend que -1 est le numel (le nombre dellement)
print(k.reshape(1,-1))
#on peux concactene 2 tensor on use la methode cat(tensor,dim=0)   by default its dim=0 
#on peux use la methode stack pour metre tous dans un array
#on use la methode stack pour cree ou represente un batch
#on a deux autre methode view et flatten 
#la methode flatten a pour parametre start_dim=x ou elle debute 

#pour faire les operation il faut avoir le meme shape il ya (+ adition , - soustration , * multiplication ,/ division ) il ya aussi des methode dans le tensor(add,sub,mul,div)
#brodcasting transfome on copy les valeur dun axe nD tensor a au (n+m)D tensor depend la trannformation  tensor de autre  tensor
#il existe la methode numpy pour transforme on numpy dans numpy il ya une methode de brodcarting qui sapelle broadcast_to(a1,a2)
#pour les operation de compareson dans le tensor il nous donne sois 0 false ou 1 true
#les type doperation que on a sont on methode (eq  == , ge >= , gt > , lt < , le <= ) avant on fesais les operation logic direct dans le tensor exemple tensor >= tensor
#on a dautre methode comme (abs ,sqrt,neg,)
#il ya element-wise component-wise point-wise


#reduction operation est une opperation qui permet de reduire les element dans un tensor 

#il ya des methode comme mean et std 
#la methode argmax donne lindice max du tensor flatten et la methode max donne la valeur max
#pour recupere la valeur on peux use la methode item et cette methode marche que avec les scalaire pour les axes on use la methode tolist ou numpy


#dataset considiration 
'''
who created the dataset?
how was the dataset created?
what transformed were used?
what intent does the dataset have?
possible unintential consequences?
is the dataset biased?
are there ethical issues with the dataset?
'''
#Mnist est une dataset de digit 
#Fashion MNIST a 10 class
'''
le process de projet est 
1)prepare la data
2)construire le model
3)train le model
4)analyse le resulta du model   
'''
#pour preapre la data il faut on va use le ETL process extract,transfom and load
import torchvision #torchvison est un package that orovides access a la populaire dataset model architecture et image transfomration
import torchvision.transforms #est une interface qui donne acces  a la pluspare (common) transformation pour image processing
#pythorch a preapre 2 class torch.utils.data.DataSet et torch.utils.data.DataLoader
#pour la dataset on dois implemente 2 methode lengeth et getitem
'''
pour la data set

pour train_set = torchvision.datasets.FashionMNIST(
root ='chemain du dataset'
,train=bool si elle va etre use pour le training
,download=bool its to dowloand la data si elle est pas presente dans le fichie
,transform=transforms.Compose([
    transforms.ToTensor()
])  ici on fait passe la composition de transformation pour mes element de la dataset
)
'''
#pour le data loader on fait just train_loader=torch.utils.data.DataLoader(tain_set)

