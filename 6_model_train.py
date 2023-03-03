import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import LSTM, Dense, Dropout
from keras import regularizers
from sklearn.metrics import classification_report, confusion_matrix,precision_score, recall_score, f1_score, roc_curve, auc
from datetime import datetime
import pandas as pd

file_name = datetime.now().strftime('output_%Y-%m-%d_%H-%M-%S.xlsx')


learning_rate = 0.001
activation_function = 'sigmoid'
epoch_no = 100
regularizer = regularizers.l1(0.01)
regularizer_out = str(regularizer)
regularizer_out = regularizer_out[regularizer_out.index('reg'):regularizer_out.index(' ', regularizer_out.index('reg'))]
loss = 'binary_crossentropy'
batch_size = 128
no_of_layers = 4
method_against_overfit = "Dropout"
no_of_lstm_units = 50




# Load the data
attacking_signal_data = np.load('attacking_signal_data.npy')
normal_signal_data = np.load('normal_signal_data.npy')

# Combine the two datasets and create labels
signal_data = np.concatenate((attacking_signal_data, normal_signal_data), axis=0)
labels = np.concatenate((np.zeros(len(attacking_signal_data)), np.ones(len(normal_signal_data))))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(signal_data, labels, test_size=0.2)



# Define the parameters for the LSTM model
timesteps = 6
num_features = 3

# Reshape the data for input into the LSTM model
#num_rows = X_train.shape[0]
#num_to_remove = num_rows % 18
#X_train = X_train[:-num_to_remove, :]
#y_train = y_train[:-num_to_remove]

#num_rows = X_test.shape[0]
#num_to_remove = num_rows % 18
#X_test = X_test[:-num_to_remove, :]
#y_test = y_test[:-num_to_remove]


#print("X_Test: ", X_test.shape)
#print("Y_Test: ",y_test.shape)
#print("X_Traint: ",X_train.shape)
#print("Y_Train: ",y_train.shape)

# Find the length of each sequence in X_train and X_test
X_train_lengths = np.array([len(x) for x in X_train])
X_test_lengths = np.array([len(x) for x in X_test])

# Determine the maximum sequence length
max_length = timesteps * num_features

# Pad the sequences with zeros to ensure that they are all of equal length
X_train = np.zeros((len(X_train), max_length), dtype='float32')
for i, x in enumerate(X_train):
    x[:len(X_train[i])] = X_train[i]

X_test = np.zeros((len(X_test), max_length), dtype='float32')
for i, x in enumerate(X_test):
    x[:len(X_test[i])] = X_test[i]

# Remove any rows that were padded
X_train = X_train[X_train_lengths <= max_length]
y_train = y_train[X_train_lengths <= max_length]
X_test = X_test[X_test_lengths <= max_length]
y_test = y_test[X_test_lengths <= max_length]



X_train = X_train.reshape(-1, timesteps, num_features)
X_test = X_test.reshape(-1, timesteps, num_features)



# First find the minimum number of samples between X_train and y_train
min_samples = min(X_train.shape[0], y_train.shape[0])

# Then reduce the number of samples in both arrays to the minimum number
X_train = X_train[:min_samples]
y_train = y_train[:min_samples]


# First find the minimum number of samples between X_test and y_test
min_samples = min(X_test.shape[0], y_test.shape[0])

# Reduce the number of samples in both arrays to the minimum number
X_test = X_test[:min_samples]
y_test = y_test[:min_samples]

print("x_train:", len(X_train))
print("x_test:", len(X_test))
print("y_train:", len(y_train))
print("y_test:", len(y_test))



# Define the LSTM model
model = Sequential()
model.add(LSTM(no_of_lstm_units, input_shape=(timesteps, num_features),return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(50))
model.add(Dropout(0.5))
model.add(Dense(1, activation=activation_function,kernel_regularizer=regularizer))

optimizer = Adam(learning_rate=learning_rate)

# Compile the model
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])


# Train the model
history = model.fit(X_train, y_train, epochs=epoch_no, batch_size=batch_size, validation_data=(X_test, y_test))


y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)
print("Y_pred:" , y_pred)# Converts probabilities to binary predictions
y_pred_prob = np.round(y_pred)
print(y_pred_prob)

# Compute ROC curve and ROC area for each class
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
print("Fpr:", fpr)
print("Tpr:", tpr)
roc_auc = auc(fpr, tpr)
print("roc_auc:", roc_auc)





# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(X_test, y_test)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


print('Accuracy:', test_acc)
print('Loss:', test_loss)
print("Precision:", precision)
print("Recall:", recall)
print("F1 score:", f1)



results = {'learning_rate': learning_rate,
           'activation_function': activation_function,
           'epoch': epoch_no,
           'regularizer': regularizer_out,
           'timestep': timesteps,
           'no_of_lstm_units': no_of_lstm_units,
           'num_features': num_features,
           'loss': loss,
           'no_of_layers': no_of_layers,
           'batch_size':batch_size,
           'method_against_overfit': method_against_overfit,
           'test_loss': test_loss,
           'test_accuracy': test_acc,
           'classification_report': classification_report(y_test,y_pred),
           'confusion_matrix': confusion_matrix(y_test, y_pred),
           'precision': precision,
            'recall': recall,
            'f1_score': f1
           }

# Plot ROC curve
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")

# Load the existing data (if any) from the file
try:
    data = pd.read_excel(file_name)
except:
    data = pd.DataFrame()

# Append the new data to the existing data
data = data.append(results, ignore_index=True)

# Save the data to the file
data.to_excel(file_name, index=False)
plt.savefig("roc_curve.png")