class_names = [
    'BENIGN', 
    'DoS Hulk', 
    'PortScan', 
    'DDoS', 
    'DoS GoldenEye', 
    'FTP-Patator', 
    'SSH-Patator', 
    'DoS slowloris', 
    'DoS Slowhttptest', 
    'Bot', 
    'Web Attack – Brute Force', 
    'Web Attack – XSS', 
    'Infiltration', 
    'Web Attack – Sql Injection', 
    'Heartbleed'
    ]

epochs = 10
hidden_dim = 128
batch_size = 1024
max_neighbors = 25
test_size = 0.2
learning_rate = 0.01
weight_decay = 0.01