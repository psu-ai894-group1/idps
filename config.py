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

feature_names = [
    'flow_duration', 'tot_fwd_pkts', 'tot_bwd_pkts',
    'totlen_fwd_pkts', 'totlen_bwd_pkts',
    'fwd_pkt_len_mean', 'bwd_pkt_len_mean',
    'flow_byts_s', 'flow_pkts_s',
    'fwd_iat_mean', 'bwd_iat_mean',
    'fin_flag_cnt', 'syn_flag_cnt', 'rst_flag_cnt', 'psh_flag_cnt',
    'ack_flag_cnt', 'urg_flag_cnt', 'cwe_flag_count', 'ece_flag_cnt',
    'down_up_ratio', 'pkt_size_avg',
]

epochs = 10
hidden_dim = 128
batch_size = 1024
max_neighbors = 25
test_size = 0.2
learning_rate = 0.01
weight_decay = 0.01