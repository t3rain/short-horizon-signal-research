SYMBOL = "BTCUSDT"
TICK_SIZE = 0.1                 # BTCUSDT perpetual price step

# all values in seconds

# Feature windows
FLOW_WINDOWS_S = (1, 5, 30)     # OFI and signed trade volume
RET_WINDOWS_S = (1, 5, 30)      # past mid-price returns

# Label settings
HORIZONS_S = (1, 5, 10, 30)     # holding times to study
PRIMARY_HORIZON_S = 10          # the one headline horizon
LATENCY_S = 1                   # order-arrival delay L
