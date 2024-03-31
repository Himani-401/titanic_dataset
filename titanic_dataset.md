```python
pip install pygame
```

    Requirement already satisfied: pygame in c:\users\welcome\anaconda3\lib\site-packages (2.5.2)Note: you may need to restart the kernel to use updated packages.
    
    


```python
pip install tk
```

    Collecting tk
      Downloading tk-0.1.0-py3-none-any.whl.metadata (693 bytes)
    Downloading tk-0.1.0-py3-none-any.whl (3.9 kB)
    Installing collected packages: tk
    Successfully installed tk-0.1.0
    Note: you may need to restart the kernel to use updated packages.
    


```python
from tkinter import *
from PIL import Image,ImageTk
```


```python
pip install mutagen
```

    Collecting mutagen
      Downloading mutagen-1.47.0-py3-none-any.whl.metadata (1.7 kB)
    Downloading mutagen-1.47.0-py3-none-any.whl (194 kB)
       ---------------------------------------- 0.0/194.4 kB ? eta -:--:--
       ------ --------------------------------- 30.7/194.4 kB ? eta -:--:--
       -------- ------------------------------ 41.0/194.4 kB 653.6 kB/s eta 0:00:01
       -------------- ------------------------ 71.7/194.4 kB 563.7 kB/s eta 0:00:01
       ---------------------------- --------- 143.4/194.4 kB 853.3 kB/s eta 0:00:01
       ------------------------------------ - 184.3/194.4 kB 857.5 kB/s eta 0:00:01
       -------------------------------------- 194.4/194.4 kB 736.6 kB/s eta 0:00:00
    Installing collected packages: mutagen
    Successfully installed mutagen-1.47.0
    Note: you may need to restart the kernel to use updated packages.
    


```python
pip install Pillow

```

    Requirement already satisfied: Pillow in c:\users\welcome\anaconda3\lib\site-packages (10.2.0)
    Note: you may need to restart the kernel to use updated packages.
    


```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


np.random.seed(42)

passenger_count = 2000

passenger_id = range(1, passenger_count + 1)
pclass = np.random.choice([1, 2, 3], size=passenger_count)
age = np.random.randint(1, 91, size=passenger_count)
sex = np.random.choice(['male', 'female'], size=passenger_count)
rate = np.random.randint(5, 500, size=passenger_count)
sibsp = np.random.randint(0, 5, size=passenger_count)
parch = np.random.randint(0, 4, size=passenger_count)
embarked = np.random.choice(['S', 'C', 'Q'], size=passenger_count)


data = pd.DataFrame({
    'PassengerId': passenger_id,
    'Pclass': pclass,
    'Age': age,
    'Sex': sex,
    'Rate': rate,
    'SibSp': sibsp,
    'Parch': parch,
    'Embarked': embarked
})


data['Survived'] = np.random.randint(0, 2, size=passenger_count)


data = pd.get_dummies(data, columns=['Sex', 'Embarked'])


X = data.drop(['PassengerId', 'Survived'], axis=1)
y = data['Survived']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LogisticRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

```

    Accuracy: 0.48
    
    Classification Report:
                  precision    recall  f1-score   support
    
               0       0.38      0.19      0.26       185
               1       0.51      0.73      0.60       215
    
        accuracy                           0.48       400
       macro avg       0.45      0.46      0.43       400
    weighted avg       0.45      0.48      0.44       400
    
    


```python

```
