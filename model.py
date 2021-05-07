# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

df = pd.read_csv('./uw-cs480-fall20/train.csv', encoding='utf-8')
df_test = pd.read_csv('./uw-cs480-fall20/test.csv', encoding='utf-8')

ids = np.unique(df['id'])
categories = np.unique(df['category'])
genders = np.unique(df['gender'])
baseColours = np.unique(df['baseColour'])
seasons = np.unique(df['season'])
usages = np.unique(df['usage'])



from sklearn.preprocessing import OneHotEncoder, LabelEncoder
# transform and map category
category_le = LabelEncoder()
category_labels = category_le.fit_transform(df['category'])
df['Category_Label'] = category_labels

# transform and map gender
gender_le = LabelEncoder()
gender_labels = gender_le.fit_transform(df['gender'])
df['Gender_Label'] = gender_labels

# transform and map baseColour
baseColour_le = LabelEncoder()
baseColour_labels = baseColour_le.fit_transform(df['baseColour'])
df['BaseColour_Label'] = baseColour_labels

# transform and map season
season_le = LabelEncoder()
season_labels = season_le.fit_transform(df['season'])
df['Season_Label'] = season_labels

# transform and map usage
usage_le = LabelEncoder()
usage_labels = usage_le.fit_transform(df['usage'])
df['Usage_Label'] = usage_labels

df_sub = df[['id','category','Category_Label','gender','Gender_Label','baseColour','BaseColour_Label','season','Season_Label','usage','Usage_Label']]
df_sub.iloc[4:10]




# encode Category using one-hot encoding scheme
category_ohe = OneHotEncoder()
category_feature_arr = category_ohe.fit_transform(
                              df[['Category_Label']]).toarray()
category_feature_labels = list(category_le.classes_)
category_features = pd.DataFrame(category_feature_arr, 
                            columns=category_feature_labels)

# encode gender using one-hot encoding scheme
gender_ohe = OneHotEncoder()
gender_feature_arr = gender_ohe.fit_transform(
                              df[['Gender_Label']]).toarray()
gender_feature_labels = list(gender_le.classes_)
gender_features = pd.DataFrame(gender_feature_arr, 
                            columns=gender_feature_labels)

# encode baseColour using one-hot encoding scheme
baseColour_ohe = OneHotEncoder()
baseColour_feature_arr = baseColour_ohe.fit_transform(
                              df[['BaseColour_Label']]).toarray()
baseColour_feature_labels = list(baseColour_le.classes_)
baseColour_features = pd.DataFrame(baseColour_feature_arr, 
                            columns=baseColour_feature_labels)

# encode season using one-hot encoding scheme
season_ohe = OneHotEncoder()
season_feature_arr = season_ohe.fit_transform(
                              df[['Season_Label']]).toarray()
season_feature_labels = list(season_le.classes_)
season_features = pd.DataFrame(season_feature_arr, 
                            columns=season_feature_labels)

# encode usage using one-hot encoding scheme
usage_ohe = OneHotEncoder()
usage_feature_arr = usage_ohe.fit_transform(
                              df[['Usage_Label']]).toarray()
usage_feature_labels = list(usage_le.classes_)
usage_features = pd.DataFrame(usage_feature_arr, 
                            columns=usage_feature_labels)
							
							
							
df_ohe = pd.concat([df_sub, category_features, gender_features, baseColour_features, season_features, usage_features], axis=1)

X = pd.concat([df['id'],gender_features, baseColour_features, season_features, usage_features], axis=1)

y = df['Category_Label']


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)


from sklearn.linear_model import LogisticRegression
logisticRegr = LogisticRegression()
logisticRegr.fit(x_train, y_train)
predictions = logisticRegr.predict(x_test)
score = logisticRegr.score(x_test, y_test)
print(score)

from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(x_train, y_train)
predictionstree = clf.predict(x_test)
scoretree = clf.score(x_test, y_test)
print(scoretree)

from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
scaling = MinMaxScaler(feature_range=(-1,1)).fit(x_train)
x_train = scaling.transform(x_train)
x_test = scaling.transform(x_test)
clf = svm.SVC(kernel='linear') # Linear Kernel
clf = clf.fit(x_train, y_train)
predictionsSVM = clf.predict(x_test)
scoreSVM = clf.score(x_test, y_test)
print(scoreSVM)



from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf = clf.fit(x_train, y_train)
predictionsRF = clf.predict(x_test)
scoreRF = clf.score(x_test, y_test)
print(scoreRF)



from sklearn.neighbors import KNeighborsClassifier
# Create KNN classifier
knn = KNeighborsClassifier(n_neighbors = 25)
# Fit the classifier to the data
knn.fit(x_train,y_train)
predictionKNN = knn.predict(x_test)
scoreKNN = knn.score(x_test, y_test)
print(scoreKNN)



from sklearn.naive_bayes import BernoulliNB
clf = BernoulliNB()
clf = clf.fit(x_train, y_train)
predictionsNB = clf.predict(x_test)
scoreNB = clf.score(x_test, y_test)
print(scoreNB)


X = pd.concat([df['noisyTextDescription']], axis=1)
y = df['Category_Label']
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)



df['noisyTextDescription1'] = df['noisyTextDescription'].str.lower()
punctuation_signs = list("?:!.,;'(%)-/+&")

for punct_sign in punctuation_signs:
    df['noisyTextDescription1'] = df['noisyTextDescription1'].str.replace(punct_sign, '')
    
numbers = list("1234567890")

for num in numbers:
    df['noisyTextDescription1'] = df['noisyTextDescription1'].str.replace(num, '')
    
df['noisyTextDescription1'] = df['noisyTextDescription1'].str.replace(r'\b\w\b','').str.replace(r'\s+', ' ')




df_test['noisyTextDescription1'] = df_test['noisyTextDescription'].str.lower()
punctuation_signs = list("?:!.,;'(%)-/+&")

for punct_sign in punctuation_signs:
    df_test['noisyTextDescription1'] = df_test['noisyTextDescription1'].str.replace(punct_sign, '')
    
numbers = list("1234567890")

for num in numbers:
    df_test['noisyTextDescription1'] = df_test['noisyTextDescription1'].str.replace(num, '')
    
df_test['noisyTextDescription1'] = df_test['noisyTextDescription1'].str.replace(r'\b\w\b','').str.replace(r'\s+', ' ')


df_test['noisyTextDescription1']

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences 
# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 50000
# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 250
# This is fixed.
EMBEDDING_DIM = 100
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df['noisyTextDescription'].values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))



X = tokenizer.texts_to_sequences(df['noisyTextDescription'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', X.shape)

Y = pd.get_dummies(df['category']).values
print('Shape of label tensor:', Y.shape)

X_train, Y_train = X, Y
print(X_train.shape,Y_train.shape)


X_test = tokenizer.texts_to_sequences(df_test['noisyTextDescription'].values)
X_test = pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', X_test.shape)

from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import LSTM, Dense, Dropout, SpatialDropout1D
from keras.callbacks import ModelCheckpoint, EarlyStopping

model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(27, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 5
batch_size = 64
print(X_train.shape,Y_train.shape)
history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])


y_pred = model.predict(X_test)

print(y_pred)


y_classes = y_pred.argmax(axis=-1)
print(y_classes)


preds = pd.DataFrame(y_classes) 
preds.head()

preds.to_csv('file1.csv') 


y_classes = y_pred.argmax(axis=-1)
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
le = LabelEncoder()
le.fit(["Accessories", "Apparel Set", "Bags", "Belts", "Bottomwear", "Cufflinks", "Dress", "Eyewear", "Flip Flops", "Fragrance", "Free Gifts", "Headwear", "Innerwear", "Jewellery", "Lips", "Loungewear and Nightwear","Makeup", "Nails","Sandal", "Saree","Scarves", "Shoes", "Socks","Ties", "Topwear", "Wallets","Watches"])
LabelEncoder()
list(le.classes_)

df_test['category'] = list(le.inverse_transform(y_classes))

ans = pd.concat([df_test['id'],df_test['category']], axis=1)	

ans.head()


ans.to_csv('file3.csv',index=False) 


import matplotlib.pyplot as plt
plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show();


import cv2
import os

def load_images_from_folder(folder):
    images = []
    file = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
       # print("fileanme",filename)
        if img is not None:
            images.append(img/255.0)
            file.append(filename)
    return images, file
folder="/kaggle/input/uw-cs480-fall20/suffled-images/shuffled-images/"

images, file = load_images_from_folder(folder)

editfile = [editfile.replace('.jpg', '') for editfile in file]

ids_int =  [int(x) for x in editfile]

df_images = pd.DataFrame(list(zip(ids_int,images,file)), columns = ['id','image','filename'])

df = pd.read_csv('/kaggle/input/uw-cs480-fall20/train.csv', encoding='utf-8')

df = df.merge(df_images, on=('id'))


df

df_test = pd.read_csv('/kaggle/input/uw-cs480-fall20/test.csv', encoding='utf-8')


df_test = df_test.merge(df_images, on=('id'))


df_test

data = pd.concat([df['filename'],df['category']], axis=1)

from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1/255., validation_split=0.2)

datagen = ImageDataGenerator(rescale=1/255.)

train_generator = datagen.flow_from_dataframe(dataframe=data, directory='./uw-cs480-fall20/suffled-images/shuffled-images/',
                                             x_col='filename',
                                             y_col='category',
                                             target_size=(80, 60),
                                             class_mode='categorical',
                                             batch_size=100,
                                             subset='training',
                                             seed=7)

validation_generator = datagen.flow_from_dataframe(dataframe=data, directory='./uw-cs480-fall20/suffled-images/shuffled-images/',
                                             x_col='filename',
                                             y_col='category',
                                             target_size=(80, 60),
                                             class_mode='categorical',
                                             batch_size=100,
                                             subset='validation',
                                             seed=7)
											 
											 
train_generator = datagen.flow_from_dataframe(dataframe=data, directory='./uw-cs480-fall20/suffled-images/shuffled-images/',
                                             x_col='filename',
                                             y_col='category',
                                             target_size=(80, 60),
                                             class_mode='categorical',
                                             batch_size=100,
                                             subset='training',
                                             seed=7)
											 
# example of using a pre-trained model as a classifier
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16

model = VGG16(include_top=False, input_shape=(80, 60, 3))

for layer in model.layers[:4]:
    layer.trainable = False
	
	
from keras.layers import Dense, Flatten
from keras.models import Model
# add new classifier layers
flat1 = Flatten()(model.layers[-1].output)
class1 = Dense(1024, activation='relu')(flat1)
output = Dense(27, activation='softmax')(class1)
# define new model
model = Model(inputs=model.inputs, outputs=output)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



train_steps = train_generator.n//train_generator.batch_size
validation_steps = validation_generator.n//validation_generator.batch_size
#history = model.fit_generator(train_generator,steps_per_epoch=train_steps, epochs=20)

history = model.fit_generator(train_generator,steps_per_epoch=train_steps, epochs=20, validation_data=validation_generator,validation_steps=validation_steps)

STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
model.evaluate_generator(generator=validation_generator,
steps=STEP_SIZE_TEST)

data_test = pd.concat([df_test['filename']], axis=1)


test_directory = '/kaggle/input/uw-cs480-fall20/suffled-images/shuffled-images/'
test_datagen = ImageDataGenerator(rescale=1/255.)
test_generator = test_datagen.flow_from_dataframe(dataframe=data_test, directory='./uw-cs480-fall20/suffled-images/shuffled-images/',
                                             x_col='filename',
                                             y_col=None,
                                             target_size=(80, 60),
                                             class_mode=None,
                                             shuffle=False,
                                             batch_size=1,
                                             seed=7)
											 
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
test_generator.reset()
pred=model.predict_generator(test_generator,steps=STEP_SIZE_TEST,verbose=1)

ans = (0.5 * y_pred) + (0.5 * pred)

print(ans)

y_classes = ans.argmax(axis=-1)
print(y_classes)
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
le = LabelEncoder()
le.fit(["Accessories", "Apparel Set", "Bags", "Belts", "Bottomwear", "Cufflinks", "Dress", "Eyewear", "Flip Flops", "Fragrance", "Free Gifts", "Headwear", "Innerwear", "Jewellery", "Lips", "Loungewear and Nightwear","Makeup", "Nails","Sandal", "Saree","Scarves", "Shoes", "Socks","Ties", "Topwear", "Wallets","Watches"])
LabelEncoder()
list(le.classes_)

df_test['category'] = list(le.inverse_transform(y_classes))

ans = pd.concat([df_test['id'],df_test['category']], axis=1)

ans.to_csv('file10.csv',index=False) 