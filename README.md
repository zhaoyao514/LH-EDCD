# LH-EDCD
This is partly the reproduction of the manuscript of "A Learning-based Hierarchical Edge Data Corruption Detection Framework in Edge Intelligence" on KDD99 and CIC-IDS2017.


### Requirements

* Python>=3.6
* Flask~=2.1.2
* matplotlib~=3.5.2
* numpy~=1.23.0
* requests~=2.25.1
* torch~=1.12.0
* scikit-learn~=1.1.1

### Run
```python
python fed_avg.py
```

Please see the arguments in options.py.

### Notes

* Please customize "your_path" in DatasetStore.py.
* Hyperparameter can be changed in options.py.