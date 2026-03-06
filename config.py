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
    # Connection info
    'dst_port',             # Destination port number

    # Flow-level stats
    'flow_duration',        # Total duration of the flow in microseconds
    'tot_fwd_pkts',         # Total packets sent in the forward direction
    'totlen_fwd_pkts',      # Total bytes sent in the forward direction
    'flow_byts_s',          # Flow bytes per second
    #'flow_pkts_s',          # Flow packets per second
    'fwd_pkts_s',           # Forward packets per second
    'bwd_pkts_s',           # Backward packets per second
    'down_up_ratio',        # Download/upload ratio (bwd packets / fwd packets)

    # Packet size statistics
    'fwd_pkt_len_mean',     # Mean forward packet size
    'fwd_pkt_len_max',      # Max forward packet size
    'bwd_pkt_len_mean',     # Mean backward packet size
    'bwd_pkt_len_max',      # Max backward packet size
    'pkt_len_mean',         # Mean packet size (both directions)
    'fwd_pkt_len_std',      # Std dev of forward packet size
    'pkt_len_std',          # Std dev of packet size (both directions)

    # Inter-arrival time (IAT)
    'fwd_iat_mean',         # Mean time between forward packets
    'bwd_iat_mean',         # Mean time between backward packets
    'fwd_iat_std',          # Std dev of time between forward packets
    'bwd_iat_std',          # Std dev of time between backward packets
    'flow_iat_mean',        # Mean time between packets (both directions)
    'flow_iat_std',         # Std dev of time between packets (both directions)

    # TCP flags
    'fin_flag_cnt',         # FIN flag count
    'syn_flag_cnt',         # SYN flag count
    'rst_flag_cnt',         # RST flag count
    'psh_flag_cnt',         # PSH flag count
    'ack_flag_cnt',         # ACK flag count
    #'urg_flag_cnt',         # URG flag count

    # TCP window and header
    'init_fwd_win_byts',    # Initial forward TCP window size in bytes
    'init_bwd_win_byts',    # Initial backward TCP window size in bytes
    'fwd_header_len',       # Total bytes used for forward packet headers

    # Bulk transfer rates
    #'fwd_blk_rate_avg',     # Average forward bulk transfer ratea
    #'bwd_blk_rate_avg',     # Average backward bulk transfer rate

    # Session activity
    'fwd_act_data_pkts',    # Forward packets with at least 1 byte of payload
    'active_mean',          # Mean time the flow was active before going idle
    'active_std',           # Std dev of active time
]

epochs = 64
hidden_dim = 32
num_hidden_layers = 2
batch_size = 1024
max_neighbors = 25
test_size = 0.10
xval_size = 0.20
learning_rate = 0.00025
weight_decay = 0.025
dropout_rate = 0.2
# None to disable gradient clipping
clipnorm = 1.0
early_stopping_patience = 10
class_reweighting = True
# Minimum probability for an attack class to be predicted; below this, defaults to BENIGN.
# 0.0 disables thresholding (pure argmax).
attack_confidence_threshold = 0.5
# 'all' to use every available feature, or a list of feature names to use
use_features = [
    'dst_port', 'flow_duration', 'tot_fwd_pkts', 'totlen_fwd_pkts', 'flow_byts_s',
    'fwd_pkts_s', 'bwd_pkts_s', 'down_up_ratio', 'fwd_pkt_len_mean',
    'fwd_pkt_len_max', 'fwd_pkt_len_std', 'bwd_pkt_len_mean', 'bwd_pkt_len_max',
    'pkt_len_mean', 'pkt_len_std', 'fwd_iat_mean', 'fwd_iat_std',
    'bwd_iat_mean', 'bwd_iat_std', 'flow_iat_mean', 'flow_iat_std',
    'fin_flag_cnt', 'syn_flag_cnt', 'rst_flag_cnt', 'psh_flag_cnt', 'ack_flag_cnt',
    'init_fwd_win_byts', 'init_bwd_win_byts', 'fwd_header_len',
    'fwd_act_data_pkts', 'active_mean', 'active_std',
]
