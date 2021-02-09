from sklearn.model_selection import train_test_split
from tools import add_noise

def MakeData(data,noise_lvl=0.2,test_split=0.2):

    X = data[0:10000]

    X_noisy, noise = add_noise(X,noise_level=noise_lvl)

    X_noisy_train, X_noisy_test, X_train, X_test = train_test_split(X_noisy, X, test_size=test_split, random_state=101)

    return X_noisy_train, X_noisy_test, X_train, X_test, noise
