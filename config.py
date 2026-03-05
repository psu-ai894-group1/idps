class_names = [
    'BENIGN', 
    #'DoS Hulk', 
    'PortScan', 
    #'DDoS', 
    #'DoS GoldenEye', 
    'FTP-Patator', 
    'SSH-Patator', 
    #'DoS slowloris', 
    #'DoS Slowhttptest', 
    #'Bot', 
    #'Web Attack – Brute Force', 
    #'Web Attack – XSS', 
    #'Infiltration', 
    #'Web Attack – Sql Injection', 
    #'Heartbleed'
    ]

# https://github.com/ahlashkari/CICFlowMeter/blob/master/ReadMe.txt
feature_names = [
    # Flow-level stats
    'flow_duration',        # Total duration of the flow in microseconds
    'tot_fwd_pkts',         # Total packets sent in the forward direction
    'totlen_fwd_pkts',      # Total bytes sent in the forward direction
    'flow_byts_s',          # Flow bytes per second
    'flow_pkts_s',          # Flow packets per second
    'down_up_ratio',        # Download/upload ratio (bwd packets / fwd packets)

    # Packet size statistics
    'fwd_pkt_len_mean',     # Mean forward packet size
    'fwd_pkt_len_max',      # Max forward packet size
    'bwd_pkt_len_mean',     # Mean backward packet size
    'bwd_pkt_len_max',      # Max backward packet size
    'pkt_len_mean',         # Mean packet size (both directions)
    'pkt_len_std',          # Std dev of packet size (both directions)

    # Inter-arrival time (IAT)
    'fwd_iat_mean',         # Mean time between forward packets
    'bwd_iat_mean',         # Mean time between backward packets
    'flow_iat_mean',        # Mean time between packets (both directions)
    'flow_iat_std',         # Std dev of time between packets (both directions)

    # TCP flags
    'fin_flag_cnt',         # FIN flag count
    'syn_flag_cnt',         # SYN flag count
    'rst_flag_cnt',         # RST flag count
    'psh_flag_cnt',         # PSH flag count
    'ack_flag_cnt',         # ACK flag count
    'urg_flag_cnt',         # URG flag count

    # TCP window and header
    'init_fwd_win_byts',    # Initial forward TCP window size in bytes
    'init_bwd_win_byts',    # Initial backward TCP window size in bytes
    'fwd_header_len',       # Total bytes used for forward packet headers

    # Bulk transfer rates
    'fwd_blk_rate_avg',     # Average forward bulk transfer rate
    'bwd_blk_rate_avg',     # Average backward bulk transfer rate

    # Session activity
    'fwd_act_data_pkts',    # Forward packets with at least 1 byte of payload
    'active_mean',          # Mean time the flow was active before going idle
    'active_std',           # Std dev of active time

    # Per-IP-pair repetition features
    'pair_flow_count',      # Number of flows sharing this (src, dst) pair
    'pair_flow_rate',       # Flows per second for this IP pair
]

epochs = 64
hidden_dim = 128
num_hidden_layers = 2
batch_size = 1024
max_neighbors = 25
test_size = 0.15
xval_size = 0.15
learning_rate = 0.001
weight_decay = 0.025
dropout_rate = 0.5
# None to disable gradient clipping
clipnorm = 1.0
early_stopping_patience = 10
class_reweighting = True
# 'all' to use every available feature, or a list of feature names to use
use_features = 'all'
pair_features = False
