# Communication tests

To run the `point_to_point` test that uses PyTorch's communication calls
directly, run
```bash
python point_to_point.py --backend gloo --master_addr localhost --rank 0 --master_port 8888 &
python point_to_point.py --backend gloo --master_addr localhost --rank 1 --master_port 8888
```

To run the `gloo_communication_handler` test that uses PipeDream's communication
library, run
```bash
python gloo_communication_handler.py --master_addr localhost --rank 0 --master_port 8888 &
python gloo_communication_handler.py --master_addr localhost --rank 1 --master_port 8888
```
