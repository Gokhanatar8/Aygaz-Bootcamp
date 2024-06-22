# Aygaz-Bootcamp

# Bu kodlar ile kütüphane, makine öğrenmesi modeli gibi ihtiyaç duyulan eklentileri import ile içe aktardık
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 

                
# veri setimizi içe aktarıp görüntüledik
vehicles = pd.read_csv("vehicles.csv")
vehicles


# Verileri görselleştirdik
vehicles.plot(kind = 'scatter', x = 'year', y = 'co2emissions')
vehicles.groupby('year')['drive'].value_counts().unstack()
vehicles.groupby('year')['drive'].value_counts().unstack().plot(kind = 'bar', 
                                                                stacked = True, 
                                                                figsize = (10, 5))

# Çalışmak istediğimiz veriler olan Front-Wheel Drive ve Compact Cars haricindeki özellikleri kaldırdık.
filtered_vehicles = vehicles[(vehicles['drive'] == 'Front-Wheel Drive') & (vehicles['class'] == 'Compact Cars')]


# filtrelenmiş veri setini görüntüledik
filtered_vehicles.plot(kind = 'scatter', x = 'year', y = 'co2emissions')



# 'year' ve 'co2emissions' sütunlarını seç
X = filtered_vehicles[['year']]
y = filtered_vehicles['co2emissions']

# Veri setini eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Lineer Regresyon modelini oluştur ve eğit
model = LinearRegression()
model.fit(X_train, y_train)

# Modelin eğitim seti üzerindeki performansını değerlendir
train_score = model.score(X_train, y_train)
print(f"Eğitim seti doğruluk oranı: {train_score:.2f}")

# Modelin test seti üzerindeki performansını değerlendir
test_score = model.score(X_test, y_test)
print(f"Test seti doğruluk oranı: {test_score:.2f}")

# Geleceğe dönük tahminler yap
future_years = np.array([[2025], [2030], [2035]])
future_predictions = model.predict(future_years)
print(f"2025, 2030 ve 2035 yılları için CO2 emisyon tahminleri: {future_predictions}")

# Sonuçları görselleştir
plt.scatter(X, y, color='blue')
plt.plot(future_years, future_predictions, color='red', linewidth=5)
plt.xlabel('Yıl')
plt.ylabel('CO2 Emisyonları')
plt.title('Yıl ve CO2 Emisyonları Arasındaki İlişki')
plt.show()

                       
                         
