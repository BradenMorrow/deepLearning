import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
import numpy as np 
import random
import pandas as pd
from sklearn import preprocessing
from collections import deque
import tensorflow as tf
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,LSTM,BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from keras_tuner.tuners import BayesianOptimization
from keras_tuner.engine.hyperparameters import HyperParameters

SEQ_LEN = 60 # minutes
FUTURE_PERIOD_PREDICT = 5 # minutes
RATIO_TO_PREDICT = 'ETHUSD' 
EPOCHS = 2
BATCH_SIZE = 64
LOG_DIR = f'models/tuning/{SEQ_LEN}SEQ-{FUTURE_PERIOD_PREDICT}PRED-{int(time.time())}'

pd.options.display.max_columns = 20

def classify(current, future):
	if float(future) > float(current):
		return 1
	else:
		return 0

def preprocess_df(df):
	df = df.drop('future', axis=1)
	for col in df.columns:
		if col != 'target':
			df[col] = df[col].pct_change()
			df.replace([np.inf, -np.inf], np.nan, inplace=True)
			df.dropna(inplace=True)
			df[col] = preprocessing.scale(df[col].values,with_mean=False)

	df.dropna(axis=0,inplace=True)

	sequential_data = []
	prev_days = deque(maxlen=SEQ_LEN)

	for i in df.values:
		prev_days.append([n for n in i[:-1]])
		if len(prev_days) == SEQ_LEN:
			sequential_data.append([np.array(prev_days), i[-1]])

	buys = []
	sells = []

	for seq,target in sequential_data:
		if target == 0:
			sells.append([seq,target])
		elif target == 1:
			buys.append([seq,target])

	lower = min(len(buys), len(sells))

	buys = buys[:lower]
	sells = sells[:lower]

	sequential_data = buys+sells

	random.shuffle(sequential_data)

	x = []
	y = []

	for seq,target in sequential_data:
		x.append(seq)
		y.append(target)

	return np.array(x), np.array(y)

print('Loading Data')

main_df = pd.DataFrame()
ratios = ['BCHUSD','BTCUSD','ETHUSD','LTCUSD','XRPUSD']

for ratio in ratios:
	dataset = f'trainingData\{ratio}.csv'
	df = pd.read_csv(dataset,names=['time','date','symbol','open','high','low','close','volume','volume_US'])
	df.rename(columns={'close':f'{ratio}_close', 'volume':f'{ratio}_volume','high':f'{ratio}_high','low':f'{ratio}_low'}, inplace = True)
	df.set_index('time', inplace=True)
	df = df[[f'{ratio}_close', f'{ratio}_volume',f'{ratio}_high',f'{ratio}_low']]
	if len(main_df) == 0:
		main_df = df
	else:
		main_df = main_df.join(df)

main_df['future'] = main_df[f'{RATIO_TO_PREDICT}_close'].shift(-FUTURE_PERIOD_PREDICT)
main_df['target'] = list(map(classify,main_df[f'{RATIO_TO_PREDICT}_close'],main_df['future']))
# print(main_df[[f'{RATIO_TO_PREDICT}_close', 'future','target']].head())

times = sorted(main_df.index.values)
last_5pct = times[-int(0.05*len(times))]

validation_main_df = main_df[(main_df.index >= last_5pct)]
main_df = main_df[(main_df.index < last_5pct)]

train_x, train_y = preprocess_df(main_df)
validation_x,validation_y = preprocess_df(main_df)

print('Data Done Loading')

def build_model(hp):
	model = Sequential()
	model.add(LSTM(hp.Int('Input_Units', 32 , 256, 32), input_shape=(train_x.shape[1:]), return_sequences=True))
	model.add(Dropout(0.2))
	model.add(BatchNormalization())

	model.add(LSTM(hp.Int(f"Layer_1_Units", 32 , 256, 32),input_shape=(train_x.shape[1:]), return_sequences=True))
	model.add(Dropout(0.2))
	model.add(BatchNormalization())

	model.add(LSTM(hp.Int("Layer_2_Units", 32 , 256, 32),input_shape=(train_x.shape[1:])))
	model.add(Dropout(0.2))
	model.add(BatchNormalization())

	model.add(Dense(hp.Int("Dense_Units", 32 , 256, 32), activation="relu6"))
	model.add(Dropout(0.2))

	model.add(Dense(2, activation='softmax'))

	opt = tf.keras.optimizers.Adam(learning_rate=0.001, decay=1e-6)

	model.compile(loss='sparse_categorical_crossentropy',
				optimizer=opt,
				metrics=['accuracy'])

	return model

tuner = BayesianOptimization(build_model,
					objective = 'val_accuracy',
					max_trials = 10,
					directory = LOG_DIR)

tuner.search(x=train_x,y=train_y,
			epochs=EPOCHS,
			batch_size=BATCH_SIZE,
			validation_data=(validation_x,validation_y))

print(tuner.get_best_hyperparameters()[0].values)
print(tuner.results_summary())
#richie = tuner.get_best_models()[0]

#richie.save("models/richieBotTuner")