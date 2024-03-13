import pandas as pd
import matplotlib.pyplot as plt

log_data = pd.read_csv('logs/lstm_layer_2_unit_64.csv')

plt.figure(figsize=(8, 5))
plt.plot(log_data['Iteration'], log_data['test_RMSE'], marker='o', linestyle='-', color='b')
plt.title('test_RMSE Over iteration')
plt.xlabel('iteratuib')
plt.ylabel('test_RMSE')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
